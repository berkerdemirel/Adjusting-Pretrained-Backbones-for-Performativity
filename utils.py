import sys
import time
import collections
import functools
import typing

import torch.utils
import timm
import torch
import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Subset, ConcatDataset

sys.path.append("../")
import tllib.vision.datasets as datasets
import tllib.vision.models as models
import tllib.normalization.ibn as ibn_models
from tllib.vision.transforms import ResizeImage
from tllib.utils.metric import accuracy
from tllib.utils.meter import AverageMeter, ProgressMeter

from timm.models.layers import trunc_normal_

from architectures import (
    ImageClassifier,
    build_transform,
    vit_base_patch16,
    NativeScalerWithGradNormCount,
    MLP,
)
import argparse
from tllib.vision.datasets.tabular_data import DATASET_NAMES as TABULAR_DATASET_NAMES
import nlp_utils
from torch.distributions.dirichlet import Dirichlet
from easydict import EasyDict
from timm.data.mixup import Mixup
from timm.loss import SoftTargetCrossEntropy
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
)


def save_and_exit(
    classifier: nn.Module,
    args: argparse.Namespace,
    arch: str,
    round_id: int,
    prior_data_x: torch.Tensor,
    prior_data_y: torch.Tensor,
):
    torch.save(
        classifier.state_dict(),
        "{}/{}_{}_temp_{}_round_{}_final.pth".format(
            args.checkpoint_path,
            arch,
            args.data,
            args.performative_temperature,
            round_id,
        ),
    )
    if args.pretraining_for_predictors:
        torch.save(
            {"x": prior_data_x, "y": prior_data_y},
            f"{args.checkpoint_path}/priors.pth",
        )


def get_optimizer_and_scheduler(
    classifier: nn.Module,
    args: argparse.Namespace,
    arch: str,
    train_loader: torch.utils.data.DataLoader,
    train_dataset: torch.utils.data.Dataset,
    original_train_dataset: torch.utils.data.Dataset,
) -> typing.Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    if args.iters_per_epoch is None or args.iters_per_epoch <= 0:
        current_iters_per_epoch = len(train_loader)
    elif args.auto_scale_iters:
        current_iters_per_epoch = int(
            args.iters_per_epoch * (len(train_dataset) / len(original_train_dataset))
        )
    else:
        current_iters_per_epoch = args.iters_per_epoch

    optimizer, lr_scheduler = None, None
    if args.no_training == False:
        if args.data in ["CivilComments", "Amazon", "AGNews"]:
            no_decay = ["bias", "LayerNorm.weight"]
            decay_params = []
            no_decay_params = []
            for names, params in classifier.named_parameters():
                if any(nd in names for nd in no_decay):
                    no_decay_params.append(params)
                else:
                    decay_params.append(params)
            params = [
                {"params": decay_params, "weight_decay": args.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ]
            optimizer = AdamW(params, lr=args.lr)

            lr_scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_training_steps=current_iters_per_epoch * args.epochs,
                num_warmup_steps=0,
            )
            lr_scheduler.step_every_batch = True
            lr_scheduler.use_metric = False
        else:
            if "resnet" in arch.lower():
                optimizer = SGD(
                    classifier.get_parameters(
                        base_lr=args.lr, backbone_lr_ratio=args.backbone_lr_ratio
                    ),
                    args.lr,
                    momentum=args.momentum,
                    weight_decay=args.weight_decay,
                    nesterov=True,
                )
                lr_scheduler = CosineAnnealingLR(
                    optimizer, args.epochs * current_iters_per_epoch
                )
            elif "mlp" in arch.lower():
                optimizer = SGD(
                    classifier.parameters(),
                    args.lr,
                    momentum=args.momentum,
                    weight_decay=args.weight_decay,
                    nesterov=True,
                )
                lr_scheduler = CosineAnnealingLR(
                    optimizer, args.epochs * current_iters_per_epoch
                )
            else:
                raise NotImplementedError

    return optimizer, lr_scheduler, current_iters_per_epoch


def get_training_priors(
    args: argparse.Namespace, num_classes: int, device: str
) -> typing.Optional[torch.Tensor]:
    if args.data == "CivilComments" or args.data == "AGNews":
        training_priors = torch.ones(num_classes, device=device) / num_classes
    elif (args.prior_predictor or args.oracle) and args.prior_path:
        training_priors = torch.load(args.prior_path)  # .to(device) + 1e-4

        # check if training priors is a dictionary
        if isinstance(training_priors, dict):
            training_priors = (
                torch.mean(torch.stack(training_priors["y"]), dim=0).to(device) + 1e-4
            )
        else:
            training_priors = training_priors.to(device) + 1e-4
        print("training priors", training_priors)
        print("training priors sum", torch.sum(training_priors))
    else:
        training_priors = None
    return training_priors


def get_criterion(
    mixup_fn: typing.Optional[Mixup], args: argparse.Namespace
) -> nn.Module:
    if mixup_fn is not None:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.0:
        criterion = LabelSmoothingCrossEntropyWithReduction(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    return criterion


def fill_models_to_dict(
    arch_dict: typing.Dict[str, typing.Dict[str, int]],
    original_train_dataset: torch.utils.data.Dataset,
    args: argparse.Namespace,
    num_classes: str,
    device: str,
) -> typing.Tuple[
    typing.Dict[str, typing.Dict[str, int]],
    typing.Optional[NativeScalerWithGradNormCount],
    typing.Optional[Mixup],
]:
    """Fill the model dictionary with the corresponding model instances.
    arch_dict: {"arch_name": {"switching_round":, "check_path":, "model": }}
    """
    mixup_fn = None
    for _arch, _dict in arch_dict.items():
        if args.data in ["CivilComments", "Amazon", "AGNews"]:
            if _arch == "distilbert-base-uncased":
                classifier = nlp_utils.DistilBertClassifier.from_pretrained(
                    _arch, num_labels=num_classes
                ).to(device)
            else:
                raise NotImplementedError
            loss_scaler = None

        elif _arch.lower() == "vit":
            tuning_config = EasyDict(
                # AdaptFormer
                ffn_adapt=args.tuning_choice == "adaptor",
                ffn_option="parallel",
                ffn_adapter_layernorm_option="none",
                ffn_adapter_init_option="lora",
                ffn_adapter_scalar="0.1",
                ffn_num=64,
                d_model=768,
                vpt_on=False,
                vpt_num=1,
            )
            classifier = vit_base_patch16(
                num_classes=num_classes,
                global_pool=False,
                drop_path_rate=0.0,
                tuning_config=tuning_config,
            )
            classifier = load_and_freeze_weights(
                classifier, args.load_path, args.tuning_choice == "finetuning"
            ).to(device)
            loss_scaler = NativeScalerWithGradNormCount()

            if args.lr is None or args.lr <= 0:  # only base_lr is specified
                args.lr = args.blr * args.batch_size / 256
                print("base lr: %.2e" % (args.blr))
            print("actual lr: %.2e" % args.lr)

            mixup_active = (
                args.mixup > 0 or args.cutmix > 0.0 or args.cutmix_minmax is not None
            )
            if mixup_active:
                print("Mixup is activated!")
                mixup_fn = Mixup(
                    mixup_alpha=args.mixup,
                    cutmix_alpha=args.cutmix,
                    cutmix_minmax=args.cutmix_minmax,
                    prob=args.mixup_prob,
                    switch_prob=args.mixup_switch_prob,
                    mode=args.mixup_mode,
                    label_smoothing=args.smoothing,
                    num_classes=num_classes,
                )

        elif "mlp" in _arch.lower():
            dict_pairs = {
                k: int(v)
                for k, v in [
                    pair.split("=") for pair in _arch.lower().split(",") if "=" in pair
                ]
            }
            classifier = MLP(
                input_dim=len(original_train_dataset.datasets[0].dataset.feature_names),
                output_dim=num_classes,
                n_layers=dict_pairs["n_layers"],
                hidden_dim=dict_pairs["hidden_dims"],
            ).to(device)
            loss_scaler = None

        else:
            print("=> using pre-trained model '{}'".format(_arch))
            backbone = get_model(_arch)
            pool_layer = nn.Identity() if args.no_pool else None

            classifier = ImageClassifier(
                backbone,
                num_classes,
                freeze_bn=args.freeze_bn,
                dropout_p=args.dropout_p,
                pool_layer=pool_layer,
            ).to(device)
            loss_scaler = None

        if _dict["check_path"]:
            classifier.load_state_dict(torch.load(_dict["check_path"]))
            print("model is loaded")

        _dict["model"] = classifier

    return arch_dict, loss_scaler, mixup_fn


def get_arch_dict(args: argparse.Namespace) -> typing.Dict[str, typing.Dict[str, int]]:
    if args.arch_rounds is not None:
        # note that the round number refers to the switching time
        arch_dict = parse_arch_string(
            args.arch_rounds
        )  # {"arch_name": {"switching_round":, "check_path":, "model": }}
        assert list(arch_dict.values())[-1]["switching_round"] < args.num_rounds

        if args.continue_check:
            paths = [path for path in args.check_path.split(";")]
            assert len(paths) == len(arch_dict)
            for i, _arch_dict in enumerate(arch_dict.values()):
                _arch_dict["check_path"] = paths[i]
        else:
            for i, _arch_dict in enumerate(arch_dict.values()):
                _arch_dict["check_path"] = None
    else:
        arch_dict = {args.arch: {"switching_round": 0, "check_path": None}}
        if args.continue_check:
            arch_dict[args.arch]["check_path"] = args.check_path
    return arch_dict


def get_initial_subpopulation_ratios(
    args: argparse.Namespace, shift_type: str, num_classes: int, num_domains: int
) -> typing.Dict[str, float]:
    if shift_type == "domain_class":
        if args.selected_subpopulation_index >= 0:
            raise NotImplementedError
        else:
            initial_subpopulation_ratios = (
                Dirichlet(
                    concentration=torch.ones(num_classes * num_domains)
                    * args.init_dirichlet_alpha
                )
                .sample()
                .reshape(num_domains, num_classes)
            )
            initial_subpopulation_ratios = {
                f"domain_{domain_id}_class_{class_id}": initial_subpopulation_ratios[
                    domain_id, class_id
                ]
                for domain_id in range(num_domains)
                for class_id in range(num_classes)
            }
    elif shift_type == "domain":
        if args.selected_subpopulation_index >= 0:
            assert args.selected_subpopulation_index < num_domains
            initial_subpopulation_ratios = {
                f"domain_{domain_id}": (
                    1.0 if domain_id == args.selected_subpopulation_index else 0
                )
                for domain_id in range(num_domains)
            }
        else:
            initial_subpopulation_ratios = Dirichlet(
                concentration=torch.ones(num_domains) * args.init_dirichlet_alpha
            ).sample()
            initial_subpopulation_ratios = {
                f"domain_{domain_id}": initial_subpopulation_ratios[domain_id]
                for domain_id in range(num_domains)
            }
    elif shift_type == "class":
        if args.selected_subpopulation_index >= 0:
            assert args.selected_subpopulation_index < num_classes
            initial_subpopulation_ratios = {
                f"class_{class_id}": (
                    1.0 if class_id == args.selected_subpopulation_index else 0
                )
                for class_id in range(num_classes)
            }
        else:
            initial_subpopulation_ratios = Dirichlet(
                concentration=torch.ones(num_classes) * args.init_dirichlet_alpha
            ).sample()
            initial_subpopulation_ratios = {
                f"class_{class_id}": initial_subpopulation_ratios[class_id]
                for class_id in range(num_classes)
            }
    else:
        raise ValueError(f"Invalid shift type: {shift_type}")

    return initial_subpopulation_ratios


def get_transforms(args: argparse.Namespace) -> typing.Tuple[nn.Module, nn.Module]:
    if args.data in ["CivilComments", "Amazon", "AGNews"]:
        train_transform = nlp_utils.get_transform(args.arch, args.max_token_length)
        val_transform = nlp_utils.get_transform(args.arch, args.max_token_length)
    elif args.data in TABULAR_DATASET_NAMES:
        train_transform = val_transform = None
    else:
        if args.arch.lower() == "vit":
            # train_transform, val_transform = get_transforms()
            train_transform, val_transform = [
                build_transform(is_train=is_train, args=args)
                for is_train in [True, False]
            ]
        else:
            train_transform = get_train_transform(
                args.train_resizing,
                random_horizontal_flip=True,
                random_color_jitter=True,
                random_gray_scale=True,
            )
            val_transform = get_val_transform(args.val_resizing)
    return train_transform, val_transform


def load_and_freeze_weights(model, checkpoint_path, full_fine_tuning):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    print("Load pre-trained checkpoint from: %s" % checkpoint_path)
    checkpoint_model = checkpoint["model"] if "model" in checkpoint else checkpoint
    state_dict = model.state_dict()
    for k in ["head.weight", "head.bias"]:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]
    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(msg)
    # manually initialize fc layer: following MoCo v3
    trunc_normal_(model.head.weight, std=0.01)
    model.head = torch.nn.Sequential(
        torch.nn.BatchNorm1d(model.head.in_features, affine=False, eps=1e-6), model.head
    )
    # freeze all but the head
    for name, p in model.named_parameters():
        if name in msg.missing_keys:
            p.requires_grad = True
        else:
            p.requires_grad = False if not full_fine_tuning else True
    for _, p in model.head.named_parameters():
        p.requires_grad = True
    return model


def freeze_unfreeze(model, req_grad, module):
    assert module in ["logit_adjustment", "backbone"]
    if module == "logit_adjustment":
        for name, p in model.logit_adjustment_module.named_parameters():
            p.requires_grad = req_grad
    elif module == "backbone":
        for name, p in model.model.named_parameters():
            p.requires_grad = req_grad


def get_model_names():
    return (
        sorted(
            name
            for name in models.__dict__
            if name.islower()
            and not name.startswith("__")
            and callable(models.__dict__[name])
        )
        + sorted(
            name
            for name in ibn_models.__dict__
            if name.islower()
            and not name.startswith("__")
            and callable(ibn_models.__dict__[name])
        )
        + timm.list_models()
    )


def get_model(model_name):
    if model_name in models.__dict__:
        # load models from tllib.vision.models
        backbone = models.__dict__[model_name](pretrained=True)
    elif model_name in ibn_models.__dict__:
        # load models (with ibn) from tllib.normalization.ibn
        backbone = ibn_models.__dict__[model_name](pretrained=True)
    else:
        # load models from pytorch-image-models
        backbone = timm.create_model(model_name, pretrained=True)
        try:
            backbone.out_features = backbone.get_classifier().in_features
            backbone.reset_classifier(0, "")
        except:
            backbone.out_features = backbone.head.in_features
            backbone.head = nn.Identity()
    return backbone


def get_dataset_names():
    return sorted(
        name
        for name in datasets.__dict__
        if not name.startswith("__") and callable(datasets.__dict__[name])
    )


class PriorPredDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        super(PriorPredDataset, self).__init__()
        self.x = x
        self.y = y

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)


class DomainPredDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        super(DomainPredDataset, self).__init__()
        self.x = x
        self.y = y

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)


class ConcatDatasetWithDomainLabel(ConcatDataset):
    """ConcatDataset with domain label"""

    def __init__(self, *args, transform=None, **kwargs):
        super(ConcatDatasetWithDomainLabel, self).__init__(*args, **kwargs)
        self.transform = transform
        self.index_to_domain_id = collections.OrderedDict()
        domain_id = 0
        start = 0
        for end in self.cumulative_sizes:
            for idx in range(start, end):
                self.index_to_domain_id[idx] = domain_id
            start = end
            domain_id += 1

    def __getitem__(self, index):
        img, target = super(ConcatDatasetWithDomainLabel, self).__getitem__(index)
        domain_id = self.index_to_domain_id[index]
        if self.transform:
            img = self.transform(img)
        return img, target, domain_id

    def num_domains(self):
        return len(self.cumulative_sizes)


class SubsetDatasetWithSampleGroup(Subset):

    def __init__(self, dataset, indices, subpopulation_indices):

        assert indices == functools.reduce(
            lambda a, b: a + b, subpopulation_indices.values()
        )

        super(SubsetDatasetWithSampleGroup, self).__init__(dataset, indices)

        self.subpopulation_indices = subpopulation_indices

        cumulative_sizes = np.cumsum(
            [
                len(_subpopulation_indices)
                for _subpopulation_indices in subpopulation_indices.values()
            ]
        )
        start = 0
        self.sample_idxes_per_domain = []
        for end in cumulative_sizes:
            idxes = [idx for idx in range(start, end)]
            self.sample_idxes_per_domain.append(idxes)
            start = end

    def sample_group(self, domain_id, size):
        indices = np.random.choice(
            self.subpopulation_indices["domain_" + str(domain_id)], size
        )
        batch = [self.dataset[index] for index in indices]
        data = torch.stack([item[0] for item in batch])
        target = torch.LongTensor([item[1] for item in batch])
        domain = torch.LongTensor([item[2] for item in batch])
        return data, target, batch


class LabelSmoothingCrossEntropyWithReduction(nn.Module):
    """NLL loss with label smoothing."""

    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropyWithReduction, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(
        self, x: torch.Tensor, target: torch.Tensor, reduction: str = "mean"
    ) -> torch.Tensor:
        assert reduction in [None, "mean", "sum"]
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        if reduction == "mean":
            return loss.mean()
        elif reduction == "sum":
            return loss.sum()
        elif reduction == None:
            return loss


def validate(val_loader, model, args, device, label_type="class") -> float:
    assert label_type in ["class", "domain"]
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    progress = ProgressMeter(
        len(val_loader), [batch_time, losses, top1], prefix="Test: "
    )

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, class_labels, domain_labels) in enumerate(val_loader):
            target = class_labels if label_type == "class" else domain_labels
            images = images.to(device)
            target = target.to(device)

            # compute output
            output = model(images)
            loss = F.cross_entropy(output, target)

            # measure accuracy and record loss
            acc1 = accuracy(output, target)[0]
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        print(" * Acc@1 {top1.avg:.3f} ".format(top1=top1))

    return top1.avg


def get_train_transform(
    resizing="default",
    random_horizontal_flip=True,
    random_color_jitter=True,
    random_gray_scale=True,
):
    """
    resizing mode:
        - default: random resized crop with scale factor(0.7, 1.0) and size 224;
        - cen.crop: take the center crop of 224;
        - res.|cen.crop: resize the image to 256 and take the center crop of size 224;
        - res: resize the image to 224;
        - res2x: resize the image to 448;
        - res.|crop: resize the image to 256 and take a random crop of size 224;
        - res.sma|crop: resize the image keeping its aspect ratio such that the
            smaller side is 256, then take a random crop of size 224;
        – inc.crop: “inception crop” from (Szegedy et al., 2015);
        – cif.crop: resize the image to 224, zero-pad it by 28 on each side, then take a random crop of size 224.
    """
    if resizing == "default":
        transform = T.RandomResizedCrop(224, scale=(0.7, 1.0))
    elif resizing == "cen.crop":
        transform = T.CenterCrop(224)
    elif resizing == "res.|cen.crop":
        transform = T.Compose([ResizeImage(256), T.CenterCrop(224)])
    elif resizing == "res":
        transform = ResizeImage(224)
    elif resizing == "res2x":
        transform = ResizeImage(448)
    elif resizing == "res.|crop":
        transform = T.Compose([T.Resize((256, 256)), T.RandomCrop(224)])
    elif resizing == "res.sma|crop":
        transform = T.Compose([T.Resize(256), T.RandomCrop(224)])
    elif resizing == "inc.crop":
        transform = T.RandomResizedCrop(224)
    elif resizing == "cif.crop":
        transform = T.Compose(
            [
                T.Resize((224, 224)),
                T.Pad(28),
                T.RandomCrop(224),
            ]
        )
    else:
        raise NotImplementedError(resizing)
    transforms = [transform]
    if random_horizontal_flip:
        transforms.append(T.RandomHorizontalFlip())
    if random_color_jitter:
        transforms.append(
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3)
        )
    if random_gray_scale:
        transforms.append(T.RandomGrayscale())
    transforms.extend(
        [
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return T.Compose(transforms)


def get_val_transform(resizing="default"):
    """
    resizing mode:
        - default: resize the image to 224;
        - res2x: resize the image to 448;
        - res.|cen.crop: resize the image to 256 and take the center crop of size 224;
    """
    if resizing == "default":
        transform = ResizeImage(224)
    elif resizing == "res2x":
        transform = ResizeImage(448)
    elif resizing == "res.|cen.crop":
        transform = T.Compose(
            [
                ResizeImage(256),
                T.CenterCrop(224),
            ]
        )
    else:
        raise NotImplementedError(resizing)
    return T.Compose(
        [
            transform,
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def count_significant_digits(reference):
    # Convert to string and ensure we are dealing with a positive number
    ref_str = str(abs(reference))
    # Find the position of the last non-zero digit
    last_non_zero_index = len(ref_str.rstrip("0"))
    # Count digits from the start of the number to the last non-zero digit, excluding the decimal point
    return (
        last_non_zero_index - 1
        if "." in ref_str[:last_non_zero_index]
        else last_non_zero_index
    )


def find_interval_index(boundaries, num):
    """
    Find the index of the interval into which the number falls based on given boundaries.

    Parameters:
    boundaries (list): A sorted list of boundaries defining the intervals.
    num (int): The integer to find the interval for.

    Returns:
    int: The index of the interval where the number falls.
    """
    assert num >= boundaries[0] and num <= boundaries[-1]

    low, high = 0, len(boundaries) - 1
    while low < high:
        mid = (low + high) // 2
        if boundaries[mid] <= num < boundaries[mid + 1]:
            return mid
        elif num < boundaries[mid]:
            high = mid
        else:
            low = mid + 1

    if num == boundaries[-1]:
        return high - 1  # Include num in the last interval

    raise ValueError


def parse_arch_string(arch_string):
    """
    Parameters:
    arch_string (str): The input string in the format "arch1:20;arch2:30".

    Returns:
    dict: A dictionary where the keys are architecture names and values are the associated numbers.
    """
    arch_dict = collections.OrderedDict()
    pairs = arch_string.split(";")  # Split the string into pairs
    for pair in pairs:
        if pair:  # Check to make sure the string isn't empty
            key, value = pair.split(":")
            arch_dict[key] = {
                "switching_round": int(value)
            }  # Convert value to integer and store in dictionary
    return arch_dict