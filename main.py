import random
import time
import warnings
import argparse
import os
from pathlib import Path
import sys
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.utils
import torch.nn.functional as F

import wandb

import utils
from utils import PriorPredDataset

from tllib.utils.data import ForeverDataIterator
from tllib.utils.metric import accuracy
from tllib.utils.meter import AverageMeter, ProgressMeter

from timm.data.mixup import Mixup
from transformers import DistilBertForSequenceClassification

from performative_util import (
    stratified_validation,
    get_subpopulation_shift_dataset,
    str2bool,
    stratified_validation_prior_pred,
    get_performative_datasets,
    get_performative_dataloaders,
    get_priors,
    test_after_shift_pre_adapt,
)

from architectures import (
    ImageClassifier,
    PriorPredictor,
    MLP,
    WarmupCosineScheduler,
)

sys.path.append("../..")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args: argparse.Namespace):

    print(args)
    run = wandb.init(
        project="performative_prediction",
        group=args.exp_group,
        name=args.exp_name,
        config=args,
    )

    folder_name = (
        args.exp_name + "_" + run.id
        if os.environ["WANDB_MODE"] == "online"
        else args.exp_name
    )  # permit overwriting test folders

    args.checkpoint_path = os.path.join(
        "./performative_prediction", args.exp_group, folder_name
    )
    Path(args.checkpoint_path).mkdir(parents=True, exist_ok=True)
    run.config.update({"checkpoint_path": args.checkpoint_path})
    data_path = os.path.join(args.data_root, args.data)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.deterministic = True
    warnings.warn(
        "You have chosen to seed training. "
        "This will turn on the CUDNN deterministic setting, "
        "which can slow down your training considerably! "
        "You may see unexpected behavior when restarting "
        "from checkpoints."
    )
    cudnn.benchmark = True
    random_state = np.random.RandomState(args.seed)

    # Data loading code
    train_transform, val_transform = utils.get_transforms(args)

    args.split_ratios = {
        key: float(val)
        for key, val in zip(["train", "val", "test"], args.split_ratios.split(","))
    }

    original_train_dataset, original_val_dataset, original_test_dataset, num_classes = (
        get_subpopulation_shift_dataset(
            dataset_name=args.data,
            root=data_path,
            download=True,
            train_transform=train_transform,
            val_transform=val_transform,
            seed=args.seed,
            split_ratios=args.split_ratios,
        )
    )

    num_domains = original_train_dataset.num_domains()

    print("original_train_dataset_size: ", len(original_train_dataset))
    print("original_val_dataset_size: ", len(original_val_dataset))
    print("original_test_dataset_size: ", len(original_test_dataset))
    print("num classes:", num_classes)

    # initial distribution
    initial_subpopulation_ratios = utils.get_initial_subpopulation_ratios(
        args, args.shift_type, num_classes, num_domains
    )

    test_val_subpopulation_accuracies = None

    args.arch_dict = utils.get_arch_dict(args)
    args.arch_dict, loss_scaler, mixup_fn = utils.fill_models_to_dict(
        args.arch_dict, original_train_dataset, args, num_classes, device
    )

    criterion = utils.get_criterion(mixup_fn, args)

    prior_data_x, prior_data_y = None, None
    if args.pretraining_for_predictors:
        prior_data_x, prior_data_y = [], []
        prior_data_y = []
        training_priors = None
    prior_predictor = None
    if args.prior_predictor:
        criterion = torch.nn.KLDivLoss(reduction="none")
        prior_predictor = PriorPredictor(num_classes).to(device)
        optimizer_prior = torch.optim.Adam(prior_predictor.parameters(), lr=1e-4)
        # to simulate epoch-like training (stores each round's inputs)
        prior_data_x, prior_data_y = [], []
        # make a weighted sum over instances
        loss_coeff = torch.ones(args.num_rounds, device=device)
        scale_coeff = 0.995

        if args.prior_pred_check:
            prior_predictor.load_state_dict(torch.load(args.prior_pred_check))
            print("Loaded prior predictor from", args.prior_pred_check)

    training_priors = utils.get_training_priors(args, num_classes, device)

    best_val_acc1 = 0.0

    round_intervals = [
        _dict["switching_round"] for _dict in args.arch_dict.values()
    ] + [args.num_rounds]
    assert round_intervals == sorted(round_intervals)

    for round_id in range(args.num_rounds + 1):
        # perform performative influences
        if round_id > 0:
            initial_subpopulation_ratios = None

        # select the model
        model_index = utils.find_interval_index(round_intervals, round_id)
        arch = list(args.arch_dict.keys())[model_index]
        classifier = list(args.arch_dict.values())[model_index]["model"]

        print(f"\n\n\nAt round {round_id}, we switch to model {arch}.\n\n\n")

        # get performative datasets
        train_dataset, val_dataset, test_dataset, sampled_subpopulation_indices = (
            get_performative_datasets(
                original_train_dataset,
                original_val_dataset,
                original_test_dataset,
                args,
                random_state,
                test_val_subpopulation_accuracies,
                initial_subpopulation_ratios,
            )
        )
        # get data loaders
        train_loader, val_loader, test_loader = get_performative_dataloaders(
            train_dataset, val_dataset, test_dataset, args
        )
        train_iter = ForeverDataIterator(train_loader)

        # define optimizer and lr scheduler
        optimizer, lr_scheduler, current_iters_per_epoch = (
            utils.get_optimizer_and_scheduler(
                classifier,
                args,
                arch,
                train_loader,
                train_dataset,
                original_train_dataset,
            )
        )

        print("Round beginning: testing...")
        # for oracle scaling and prior predictor training
        test_priors = get_priors(test_loader, num_classes, device)
        train_priors = get_priors(train_loader, num_classes, device)

        # testing out of distribution pre adaptation
        test_acc, test_subpopulation_accuracies, oracle_acc, prev_round_accs = (
            test_after_shift_pre_adapt(
                round_id,
                test_loader,
                classifier,
                device,
                num_classes,
                num_domains,
                args,
                sampled_subpopulation_indices,
                training_priors,
                test_priors,
                test_val_subpopulation_accuracies,
                prior_predictor,
            )
        )

        print("Round beginning: test acc on test set = {}".format(test_acc))
        print(
            f"Round beginning: subpopulation accuracy: max {max(test_subpopulation_accuracies.items(), key=lambda x: x[1])} and min {min(test_subpopulation_accuracies.items(), key=lambda x: x[1])}."
        )
        if round_id % args.log_every_n_rounds == 0:
            run.log(
                {
                    "test_acc_out_of_distribution_pre_round": (
                        test_acc * 100 if not args.oracle else oracle_acc * 100
                    ),
                    "round": round_id,
                    "acc_over_rounds": test_acc,
                    "worst_acc_over_rounds": min(
                        test_subpopulation_accuracies.values()
                    ),
                },
            )

        if round_id == args.num_rounds:
            utils.save_and_exit(
                classifier, args, arch, round_id, prior_data_x, prior_data_y
            )
            break

        model = classifier

        if args.no_training == False:
            for epoch in range(args.epochs):
                # train
                train(
                    train_iter=train_iter,
                    model=classifier,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    epoch=epoch,
                    iters_per_epoch=current_iters_per_epoch,
                    print_freq=args.print_freq,
                    round=round_id,
                    loss_scaler=loss_scaler,
                    criterion=criterion,
                    mixup_fn=mixup_fn,
                )
                if epoch % args.eval_freq == 0 or epoch == args.epochs - 1:
                    # eval
                    print(
                        f"Round {round_id} Epoch {epoch}, evaluation on validation set..."
                    )
                    best_val_acc1 = eval(
                        val_loader,
                        model,
                        args,
                        device,
                        round_id,
                        epoch,
                        arch,
                        best_val_acc1,
                    )
                    # test
                    print(f"Round {round_id} Epoch {epoch}, evaluation on test set...")
                    test_acc, test_subpopulation_accuracies = stratified_validation(
                        test_loader,
                        classifier,
                        device,
                        num_classes=num_classes,
                        num_domains=num_domains,
                        stratified_category=args.shift_type,
                        print_freq=args.print_freq,
                    )
                    print(
                        " * Worst Group Acc@1",
                        min(test_subpopulation_accuracies.values()),
                        "with worst group",
                        min(
                            test_subpopulation_accuracies,
                            key=test_subpopulation_accuracies.get,
                        ),
                    )
        else:
            print("not training")

        if (args.prior_predictor or args.pretraining_for_predictors) and round_id > 0:
            # x -> prev_round_accs
            # y -> train_priors
            # store the data
            prior_data_x.append(prev_round_accs.float())
            prior_data_y.append(train_priors)

            if args.prior_predictor:
                if args.train_prior_predictor:
                    total_loss = train_prior_predictor(
                        prior_data_x,
                        prior_data_y,
                        prior_predictor,
                        loss_coeff,
                        scale_coeff,
                        optimizer_prior,
                        criterion,
                    )
                    print("round: ", round_id, "loss:", total_loss.item())
                    run.log(
                        {
                            "loss_over_rounds": total_loss,
                        },
                        commit=False,
                    )
                test_acc_val, test_val_subpopulation_accuracies = eval_prior_predictor(
                    prior_predictor,
                    prev_round_accs,
                    test_loader,
                    classifier,
                    device,
                    num_classes,
                    num_domains,
                    args,
                    training_priors,
                    test_priors,
                )

        test_acc_val = test_acc
        test_val_subpopulation_accuracies = test_subpopulation_accuracies

        # if round_id % args.log_every_n_rounds == 0:
        # evaluate on test set
        print("test acc on test set = {}".format(test_acc_val))
        print(
            f"subpopulation accuracy: max {max(test_val_subpopulation_accuracies.items(), key=lambda x: x[1])} and min {min(test_val_subpopulation_accuracies.items(), key=lambda x: x[1])}."
        )
        run.log(
            {
                "best_test_acc_val_post_round": test_acc_val,
                "round": round_id,
                "acc_over_rounds": test_acc_val,
                "worst_acc_over_rounds": 100
                * min(test_subpopulation_accuracies.values()),
            },
            commit=False,
        )


def train(
    train_iter: ForeverDataIterator,
    model: ImageClassifier,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    loss_scaler: torch.cuda.amp.GradScaler,
    mixup_fn: Mixup,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    round: int,
    iters_per_epoch: int,
    print_freq: int = 1,
    label_type: str = "class",
):
    assert label_type in ["class", "domain"]

    batch_time = AverageMeter("Time", ":4.2f")
    data_time = AverageMeter("Data", ":3.1f")
    cls_losses = AverageMeter(f"{label_type} Loss", ":3.2f")
    cls_accs = AverageMeter(f"{label_type} Acc", ":3.1f")

    progress = ProgressMeter(
        iters_per_epoch,
        [batch_time, data_time, cls_losses, cls_accs],
        prefix="Round: [{}] Epoch: [{}]".format(round, epoch),
    )

    # switch to train mode
    model.train()

    end = time.time()
    for i in range(iters_per_epoch):
        x, class_labels, domain_labels = next(train_iter)
        x = x.to(device)
        labels = (
            class_labels.to(device)
            if label_type == "class"
            else domain_labels.to(device)
        )

        if mixup_fn is not None:
            x, labels = mixup_fn(x, labels)

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        if isinstance(model, DistilBertForSequenceClassification) or isinstance(
            model, MLP
        ):
            y = model(x)
        else:
            y, _ = model(x)

        loss = criterion(y, labels)

        cls_acc = accuracy(y, labels)[0]
        cls_accs.update(cls_acc.item(), x.size(0))
        cls_losses.update(loss.item(), x.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()

        if loss_scaler is None:
            loss.backward()
            optimizer.step()
        else:
            loss_scaler(loss, optimizer, parameters=model.parameters())

        if isinstance(lr_scheduler, WarmupCosineScheduler):
            lr_scheduler.step(i / iters_per_epoch + epoch)
        else:
            lr_scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            progress.display(i)


def eval(
    val_loader: torch.utils.data.DataLoader,
    model: ImageClassifier,
    args: argparse.Namespace,
    device: str,
    round_id: int,
    epoch: int,
    arch: str,
    best_val_acc1: float,
):
    best_model_filename = None
    # evaluate on validation set
    acc1 = utils.validate(val_loader, model, args, device)

    # remember best acc@1 and save checkpoint
    if acc1 > best_val_acc1:
        # If there's a previously saved best model, delete it
        if best_model_filename and os.path.exists(best_model_filename):
            os.remove(best_model_filename)

        # Construct the new best model's filename
        best_model_filename = "{}/{}_{}_temp_{}_round_{}_epoch_{}.pth".format(
            args.checkpoint_path,
            arch,
            args.data,
            args.performative_temperature,
            round_id,
            epoch,
        )
        # Save the new best model
        torch.save(model.state_dict(), best_model_filename)
        best_val_acc1 = max(acc1, best_val_acc1)
    return best_val_acc1


def train_prior_predictor(
    prior_data_x: List,
    prior_data_y: List,
    prior_predictor: PriorPredictor,
    loss_coeff: torch.Tensor,
    scale_coeff: float,
    optimizer_prior: torch.optim.Optimizer,
    criterion: torch.nn.Module,
) -> torch.Tensor:
    # create a dataset
    prior_dataset = PriorPredDataset(prior_data_x, prior_data_y)
    b_size = 1 if len(prior_data_x) // 2 == 0 else len(prior_data_x) // 2
    prior_dataloader = torch.utils.data.DataLoader(
        prior_dataset, batch_size=b_size, shuffle=True
    )

    total_loss = 0
    for i, (x, y) in enumerate(prior_dataloader):
        x, y = x.to(device), y.to(device)
        # B, D = x.shape
        out = prior_predictor(x)
        # priors = F.softmax(out, dim=1)
        # KL-div loss
        log_priors = F.log_softmax(out, dim=1)
        loss = criterion(log_priors.float(), y.float())  # no reduction, outputs BxC
        loss = torch.mean(loss, dim=1)  # per sample loss of size B
        loss = torch.mean(torch.multiply(loss, loss_coeff[0 : loss.shape[0]])).view(
            1
        )  # scale it with loss coeffs and sum

        optimizer_prior.zero_grad()
        loss.backward()
        optimizer_prior.step()

        total_loss += loss
        loss_coeff[0 : loss.shape[0]] = (
            loss_coeff[0 : loss.shape[0]] * scale_coeff
        )  # update loss coeffs

    return total_loss


def eval_prior_predictor(
    prior_predictor: PriorPredictor,
    prev_round_accs: torch.Tensor,
    test_loader: torch.utils.data.DataLoader,
    classifier: ImageClassifier,
    device: str,
    num_classes: int,
    num_domains: int,
    args: argparse.Namespace,
    training_priors: torch.Tensor,
    test_priors: torch.Tensor,
) -> Tuple[float, Dict[str, float]]:
    with torch.inference_mode():  # do current round's prediction
        out = prior_predictor(prev_round_accs.float())
        priors = F.softmax(out, dim=0)
        print("-------------------")
        print("pred", priors)
        print("gt", test_priors)
        print("-------------------")
        print()
    test_acc_val, test_val_subpopulation_accuracies = stratified_validation_prior_pred(
        test_loader,
        classifier,
        device,
        num_classes=num_classes,
        num_domains=num_domains,
        stratified_category=args.shift_type,
        print_freq=args.print_freq,
        training_priors=training_priors,
        eval_priors=priors,
    )
    return test_acc_val, test_val_subpopulation_accuracies


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline for Domain Generalization")
    # dataset parameters
    parser.add_argument(
        "--root", default=".", metavar="DIR", help="root path of dataset"
    )
    parser.add_argument("--data_root", metavar="DIR", help="root path of dataset")
    parser.add_argument(
        "-d",
        "--data",
        metavar="DATA",
        default="PACS",
        help="dataset: " + " | ".join(utils.get_dataset_names()) + " (default: PACS)",
    )
    parser.add_argument("--train-resizing", type=str, default="default")
    parser.add_argument("--val-resizing", type=str, default="default")
    parser.add_argument("--eval_freq", type=int, default=1)
    parser.add_argument("--log_every_n_rounds", type=int, default=10)
    parser.add_argument("--max_token_length", type=int, default=300)

    # model parameters
    parser.add_argument("-a", "--arch", metavar="ARCH", default="resnet50")
    parser.add_argument(
        "--no-pool",
        action="store_true",
        help="no pool layer after the feature extractor.",
    )
    parser.add_argument(
        "--freeze-bn", action="store_true", help="whether freeze all bn layers"
    )
    parser.add_argument(
        "--dropout-p",
        type=float,
        default=0.1,
        help="only activated when freeze-bn is True",
    )
    parser.add_argument("--load_path", type=str, default=None)

    parser.add_argument("--arch_rounds", type=str, default=None)

    # training parameters
    parser.add_argument(
        "-b",
        "--batch_size",
        default=36,
        type=int,
        metavar="N",
        help="mini-batch size (default: 36)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        metavar="LR",
        help="learning rate (absolute lr)",
    )
    parser.add_argument(
        "--blr",
        type=float,
        default=0.1,
        metavar="LR",
        help="base learning rate: absolute_lr = base_lr * total_batch_size / 256",
    )
    parser.add_argument(
        "--backbone_lr_ratio",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--momentum", default=0.9, type=float, metavar="M", help="momentum"
    )
    parser.add_argument(
        "--weight_decay",
        default=0.0005,
        type=float,
        metavar="W",
        help="weight decay (default: 5e-4)",
    )
    parser.add_argument(
        "--workers",
        default=4,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 4)",
    )
    parser.add_argument(
        "--val_workers",
        default=4,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 4)",
    )
    parser.add_argument(
        "--epochs",
        default=20,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "--iters_per_epoch",
        default=None,
        type=int,
        help="Number of iterations per epoch",
    )
    parser.add_argument(
        "-p",
        "--print_freq",
        default=10,
        type=int,
        metavar="N",
        help="print frequency (default: 100)",
    )
    parser.add_argument(
        "--seed", default=0, type=int, help="seed for initializing training. "
    )
    parser.add_argument("--auto_scale_iters", default=True, type=str2bool)
    parser.add_argument(
        "--tuning_choice",
        type=str,
        default="finetuning",
        choices=["adaptor", "linear_probing", "finetuning"],
    )

    # logging
    parser.add_argument("--exp_group", type=str)
    parser.add_argument("--exp_name", type=str)

    # performative prediction
    parser.add_argument("--num_rounds", type=int, default=1)
    parser.add_argument("--performative_temperature", type=float, default=1)
    parser.add_argument("--positive_correlation", type=str2bool, default=False)

    parser.add_argument("--split_ratios", type=str, default="0.4,0.3,0.3")

    parser.add_argument("--shift_type", type=str, default="domain_class")
    parser.add_argument("--init_dirichlet_alpha", type=float, default=100)
    parser.add_argument("--base_size", type=int, default=-1)
    parser.add_argument("--test_base_size", type=int, default=-1)

    parser.add_argument("--no_training", default=False, type=str2bool)
    parser.add_argument("--save_check", type=str2bool, default="False")
    parser.add_argument("--continue_check", type=str2bool, default="False")
    parser.add_argument("--prior_predictor", type=str2bool, default="False")
    parser.add_argument("--oracle", type=str2bool, default="False")
    parser.add_argument("--prior_path", type=str)
    parser.add_argument("--check_path", type=str)
    parser.add_argument("--pretraining_for_predictors", type=str2bool, default=False)
    parser.add_argument("--full_covariate_shift", type=str2bool, default=False)
    parser.add_argument("--prior_pred_check", type=str, default=None)
    parser.add_argument("--train_prior_predictor", type=str, default="True")

    parser.add_argument("--selected_subpopulation_index", type=int, default=-1)
    # mixup
    parser.add_argument(
        "--mixup", type=float, default=0, help="mixup alpha, mixup enabled if > 0."
    )
    parser.add_argument(
        "--cutmix", type=float, default=0, help="cutmix alpha, cutmix enabled if > 0."
    )
    parser.add_argument(
        "--cutmix_minmax",
        type=float,
        nargs="+",
        default=None,
        help="cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)",
    )
    parser.add_argument(
        "--mixup_prob",
        type=float,
        default=1.0,
        help="Probability of performing mixup or cutmix when either/both is enabled",
    )
    parser.add_argument(
        "--mixup_switch_prob",
        type=float,
        default=0.5,
        help="Probability of switching to cutmix when both mixup and cutmix enabled",
    )
    parser.add_argument(
        "--mixup_mode",
        type=str,
        default="batch",
        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"',
    )

    parser.add_argument("--input_size", default=224, type=int, help="images input size")
    parser.add_argument(
        "--drop_path",
        type=float,
        default=0,
        metavar="PCT",
        help="Drop path rate (default: 0.1)",
    )

    # Optimizer parameters
    parser.add_argument(
        "--clip_grad",
        type=float,
        default=None,
        metavar="NORM",
        help="Clip gradient norm (default: None, no clipping)",
    )
    parser.add_argument(
        "--layer_decay",
        type=float,
        default=0.75,
        help="layer-wise lr decay from ELECTRA/BEiT",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=0,
        metavar="LR",
        help="lower lr bound for cyclic schedulers that hit 0",
    )
    parser.add_argument(
        "--warmup_epochs", type=int, metavar="N", help="epochs to warmup LR"
    )

    # # Augmentation parameters
    parser.add_argument(
        "--color_jitter",
        type=float,
        default=None,
        metavar="PCT",
        help="Color jitter factor (enabled only when not using Auto/RandAug)",
    )
    parser.add_argument(
        "--aa",
        type=str,
        default="rand-m9-mstd0.5-inc1",
        metavar="NAME",
        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)',
    ),
    parser.add_argument(
        "--smoothing", type=float, default=0.1, help="Label smoothing (default: 0.1)"
    )

    # * Random Erase params
    parser.add_argument(
        "--reprob",
        type=float,
        default=0.25,
        metavar="PCT",
        help="Random erase prob (default: 0.25)",
    )
    parser.add_argument(
        "--remode",
        type=str,
        default="pixel",
        help='Random erase mode (default: "pixel")',
    )
    parser.add_argument(
        "--recount", type=int, default=1, help="Random erase count (default: 1)"
    )
    parser.add_argument(
        "--resplit",
        action="store_true",
        default=False,
        help="Do not random erase first (clean) augmentation split",
    )

    args = parser.parse_args()

    main(args)
