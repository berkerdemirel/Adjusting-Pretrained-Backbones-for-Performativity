import os
import time

import functools
from collections import OrderedDict
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Subset

from zipfile import ZipFile
import tarfile
import gdown
import json
import shutil
import pandas as pd
import ast

from utils import (
    ConcatDatasetWithDomainLabel,
    SubsetDatasetWithSampleGroup,
    count_significant_digits,
)
from tllib.utils.metric import accuracy
from tllib.utils.meter import AverageMeter, ProgressMeter
import tllib.vision.datasets as datasets
from tllib.vision.datasets.tabular_data import DATASET_NAMES as TABULAR_DATASET_NAMES

from wilds.datasets.wilds_dataset import WILDSSubset

import argparse
from timm.data.loader import MultiEpochsDataLoader
from torch.utils.data import DataLoader
import collections


def partition_subpopulation_shift_dataset(
    dataset: ConcatDatasetWithDomainLabel,
    subpopulation_ratios: dict,
    random_state: np.random.RandomState,
    shift_type: str = "domain_class",
    base_size: int = -1,
    is_train_set: bool = False,
    full_covariate_shift: bool = False,
):
    """
    obtain an index dictionary consisting subpopulations, with the index [domain_index, class_index].
    we then select a subset of these indices (globally) to create a subset.
    """
    domain_ids = None
    if full_covariate_shift:
        # only show half of the domains randomly
        domain_ids = random_state.choice(
            np.arange(dataset.num_domains()), size=2, replace=False
        )

    index_subpopulations = get_subpopulation_indices(
        stratified_category=shift_type,
        concatenated_dataset=dataset,
        full_covariate_shift=full_covariate_shift,
        domain_ids=domain_ids,
    )

    if base_size <= 0:
        selected_population_index = next(
            (key for key, value in subpopulation_ratios.items() if value == 1), None
        )
        base_size = (
            len(index_subpopulations[selected_population_index])
            if selected_population_index
            else len(dataset)
        )

    cnt = 0
    pdf = list(subpopulation_ratios.values())
    pdf /= np.sum(pdf)  # for numerical consistency

    key_list = list(subpopulation_ratios.keys())
    sampled_subpopulation_indices = OrderedDict(
        {key: [] for key in index_subpopulations.keys()}
    )

    if not is_train_set:
        # add 5 samples to each subpopulation
        for i in range(len(key_list)):
            sampled_subpopulation_indices[key_list[i]] = random_state.choice(
                index_subpopulations[key_list[i]], size=5, replace=False
            ).tolist()
            cnt += len(sampled_subpopulation_indices[key_list[i]])

    while cnt < base_size:
        sampled_subpopulation_key = random_state.choice(key_list, p=pdf)
        if (
            sampled_subpopulation_key not in index_subpopulations
            or len(index_subpopulations[sampled_subpopulation_key]) == 0
        ):
            continue
        if len(index_subpopulations[sampled_subpopulation_key]) == len(
            sampled_subpopulation_indices[sampled_subpopulation_key]
        ):  # if we exhausted the subpopulation
            # to optimize sampling process, we set the probability of exhausted subpopulation to be 1e-4
            exhausted_id = int(
                sampled_subpopulation_key[sampled_subpopulation_key.find("_") + 1 :]
            )
            pdf[exhausted_id] = 1e-4
            pdf /= np.sum(pdf)
            continue
        sample_id = random_state.choice(
            list(
                set(index_subpopulations[sampled_subpopulation_key])
                - set(sampled_subpopulation_indices[sampled_subpopulation_key])
            )
        )
        sampled_subpopulation_indices[sampled_subpopulation_key].append(sample_id)
        cnt += 1

    print(
        f"Subpopulation sizes: {[len(sub) for sub in sampled_subpopulation_indices.values()]}"
    )
    print(
        "Sum is {}".format(
            np.sum([len(sub) for sub in sampled_subpopulation_indices.values()])
        )
    )

    return sampled_subpopulation_indices


def get_subpopulation_ratios(
    subpopulation_accuracies: dict,
    temperature: float,
    positive_correlation: bool,
):
    assert temperature > 0
    temperature = temperature if positive_correlation else -temperature
    subpopulation_accuracies = OrderedDict(subpopulation_accuracies)
    subpopulation_ratios = softmax_with_temperature(
        logits=torch.tensor(list(subpopulation_accuracies.values())),
        temperature=temperature,
    )
    n_digits = count_significant_digits(1 / len(subpopulation_ratios)) + 2
    return OrderedDict(
        {
            list(subpopulation_accuracies.keys())[index]: round(
                subpopulation_ratio.item(), n_digits
            )
            for index, subpopulation_ratio in enumerate(subpopulation_ratios)
        }
    )


def softmax_with_temperature(logits, temperature=1.0):
    """
    Compute the softmax of the input logits with a temperature parameter.

    Args:
        logits (torch.Tensor): Input logits.
        temperature (float): Temperature parameter. Higher values (e.g., > 1.0) make the distribution
                            more uniform, while lower values (e.g., < 1.0) sharpen the distribution.

    Returns:
        torch.Tensor: Softmax probabilities.
    """
    if abs(temperature) == 1.0:
        # Standard softmax
        return torch.nn.functional.softmax(logits / temperature, dim=-1)
    else:
        logits = logits / temperature
        max_logits = torch.max(logits, dim=-1, keepdim=True)[0]
        exp_logits = torch.exp(logits - max_logits)
        softmax_probs = exp_logits / exp_logits.sum(dim=-1, keepdim=True)
        return softmax_probs


def stratified_validation(
    val_loader,
    model,
    device,
    num_domains,
    num_classes,
    stratified_category,
    print_freq,
    oracle=False,
    training_priors=None,
    test_priors=None,
) -> float:
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    progress = ProgressMeter(
        len(val_loader), [batch_time, losses, top1], prefix="Test: "
    )

    assert stratified_category in ["class", "domain", "domain_class"]
    if stratified_category == "domain_class":
        subpopulation_records = OrderedDict(
            {
                f"domain_{domain_label}_class_{class_label}": []
                for domain_label in range(num_domains)
                for class_label in range(num_classes)
            }
        )
    elif stratified_category == "domain":
        subpopulation_records = OrderedDict(
            {f"domain_{domain_label}": [] for domain_label in range(num_domains)}
        )
    else:
        subpopulation_records = OrderedDict(
            {f"class_{class_label}": [] for class_label in range(num_classes)}
        )

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()

        for i, (images, target, domain_ids) in enumerate(val_loader):

            images = images.to(device)
            target = target.to(device)

            # compute output
            output = model(images)
            loss = F.cross_entropy(output, target)
            # measure accuracy and record loss
            if oracle == True:
                output = F.softmax(output, dim=1)
                output = torch.divide(output, training_priors)
                output = torch.multiply(output, test_priors)
            acc1 = accuracy(output, target)[0]

            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))

            corrects = get_corrects(outputs=output, targets=target)
            for index, correct in enumerate(corrects):
                if stratified_category == "domain_class":
                    subpopulation_records[
                        f"domain_{domain_ids[index].item()}_class_{target[index].item()}"
                    ].append(correct)
                elif stratified_category == "domain":
                    subpopulation_records[f"domain_{domain_ids[index].item()}"].append(
                        correct
                    )
                else:
                    subpopulation_records[f"class_{target[index].item()}"].append(
                        correct
                    )

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                progress.display(i)

        print(" * Acc@1 {top1.avg:.3f} ".format(top1=top1))

    subpopulation_accuracies = OrderedDict(
        {
            key: (sum(record) / len(record) if len(record) > 0 else 0)
            for key, record in subpopulation_records.items()
        }
    )
    return top1.avg, subpopulation_accuracies


def stratified_validation_prior_pred(
    val_loader,
    model,
    device,
    num_domains,
    num_classes,
    stratified_category,
    print_freq,
    training_priors,
    eval_priors,
) -> float:
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    progress = ProgressMeter(
        len(val_loader), [batch_time, losses, top1], prefix="Test: "
    )

    assert stratified_category in ["class", "domain", "domain_class"]
    if stratified_category == "domain_class":
        subpopulation_records = OrderedDict(
            {
                f"domain_{domain_label}_class_{class_label}": []
                for domain_label in range(num_domains)
                for class_label in range(num_classes)
            }
        )
    elif stratified_category == "domain":
        subpopulation_records = OrderedDict(
            {f"domain_{domain_label}": [] for domain_label in range(num_domains)}
        )
    else:
        subpopulation_records = OrderedDict(
            {f"class_{class_label}": [] for class_label in range(num_classes)}
        )

    # switch to evaluate mode
    model.eval()

    with torch.inference_mode():
        end = time.time()

        for i, (images, target, domain_ids) in enumerate(val_loader):

            images = images.to(device)
            target = target.to(device)
            # compute output
            output = model(images)
            # measure accuracy and record loss
            output = F.softmax(output, dim=1)
            output = torch.divide(output, training_priors)
            output = torch.multiply(output, eval_priors)

            loss = F.cross_entropy(output, target)

            acc1 = accuracy(output, target)[0]

            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))

            corrects = get_corrects(outputs=output, targets=target)
            for index, correct in enumerate(corrects):
                if stratified_category == "domain_class":
                    subpopulation_records[
                        f"domain_{domain_ids[index].item()}_class_{target[index].item()}"
                    ].append(correct)
                elif stratified_category == "domain":
                    subpopulation_records[f"domain_{domain_ids[index].item()}"].append(
                        correct
                    )
                else:
                    subpopulation_records[f"class_{target[index].item()}"].append(
                        correct
                    )

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                progress.display(i)

        print(" * Acc@1 {top1.avg:.3f} ".format(top1=top1))

    subpopulation_accuracies = OrderedDict(
        {
            key: (sum(record) / len(record) if len(record) > 0 else 0)
            for key, record in subpopulation_records.items()
        }
    )
    return top1.avg, subpopulation_accuracies


def get_corrects(outputs, targets):
    with torch.no_grad():
        _, preds = outputs.topk(1, 1, True, True)
        preds = preds.t()
        corrects = preds.eq(targets[None])

    return corrects.tolist()[0]


def get_priors(dataloader, num_classes, device):
    labels = torch.cat([label for _, label, _ in dataloader]).tolist()
    freqs = collections.Counter(labels)
    priors = np.array([freqs[i] for i in range(num_classes)]).astype(float)
    priors /= np.sum(priors)
    priors = torch.from_numpy(priors).to(device)
    return priors


def test_after_shift_pre_adapt(
    round_id: int,
    test_loader: torch.utils.data.DataLoader,
    classifier: torch.nn.Module,
    device: str,
    num_classes: int,
    num_domains: int,
    args: argparse.Namespace,
    sampled_subpopulation_indices: Dict[str, List[int]],
    training_priors: Optional[torch.Tensor],
    test_priors: Optional[torch.Tensor],
    test_val_subpopulation_accuracies: Optional[Dict[str, float]],
    prior_predictor: Optional[torch.nn.Module],
):
    prev_round_accs = None
    oracle_acc = None
    if (args.prior_predictor or args.pretraining_for_predictors):
        if test_val_subpopulation_accuracies != None:
            prev_round_accs = torch.Tensor(
                list(test_val_subpopulation_accuracies.values())
            ).to(device)
        else:
            prev_round_accs = test_val_subpopulation_accuracies

    # acc_t-1 -> priors
    if args.prior_predictor and round_id != 0:
        with torch.inference_mode():
            out = prior_predictor(prev_round_accs)
            priors = F.softmax(out, dim=0)

        test_acc, test_subpopulation_accuracies = stratified_validation_prior_pred(
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
    else:
        test_acc, test_subpopulation_accuracies = stratified_validation(
            test_loader,
            classifier,
            device,
            num_classes=num_classes,
            num_domains=num_domains,
            stratified_category=args.shift_type,
            print_freq=args.print_freq,
            oracle=False,
            training_priors=None,
            test_priors=None,
        )

    if args.oracle:
        oracle_acc, _ = stratified_validation(
            test_loader,
            classifier,
            device,
            num_classes=num_classes,
            num_domains=num_domains,
            stratified_category=args.shift_type,
            print_freq=args.print_freq,
            oracle=args.oracle,
            training_priors=training_priors,
            test_priors=test_priors,
        )

    non_zero_values = [
        value for value in test_subpopulation_accuracies.values() if value != 0
    ]
    mean_non_zero = (
        sum(non_zero_values) / len(non_zero_values) if non_zero_values else 0
    )
    test_subpopulation_accuracies = collections.OrderedDict(
        (
            k,
            (
                mean_non_zero
                if v == 0
                and len(sampled_subpopulation_indices["class_{}".format(i)]) == 0
                else v
            ),
        )
        for i, (k, v) in enumerate(test_subpopulation_accuracies.items())
    )
    return test_acc, test_subpopulation_accuracies, oracle_acc, prev_round_accs


def get_performative_dataloaders(
    train_dataset: SubsetDatasetWithSampleGroup,
    val_dataset: SubsetDatasetWithSampleGroup,
    test_dataset: SubsetDatasetWithSampleGroup,
    args: argparse.Namespace,
) -> Tuple[
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
]:
    train_loader = MultiEpochsDataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        drop_last=True,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.val_workers,
        pin_memory=True,
    )
    if not args.no_training:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.val_workers,
            pin_memory=True,
        )
    else:
        val_loader = test_loader

    return train_loader, val_loader, test_loader


def get_performative_datasets(
    original_train_dataset: ConcatDatasetWithDomainLabel,
    original_val_dataset: ConcatDatasetWithDomainLabel,
    original_test_dataset: ConcatDatasetWithDomainLabel,
    args: argparse.Namespace,
    random_state: np.random.RandomState,
    test_val_subpopulation_accuracies: Dict[str, float],
    initial_subpopulation_ratios: Dict[str, float],
) -> Tuple[
    SubsetDatasetWithSampleGroup,
    SubsetDatasetWithSampleGroup,
    SubsetDatasetWithSampleGroup,
    List,
]:
    datasets = {}
    for original_dataset, key in zip(
        [original_val_dataset, original_test_dataset, original_train_dataset],
        ["val", "test", "train"],
    ):
        if args.no_training and key == "val":
            datasets["val"] = []
            continue
        datasets[key], sampled_subpopulation_indices = performative_shift(
            original_dataset,
            temperature=args.performative_temperature,
            positive_correlation=args.positive_correlation,
            shift_type=args.shift_type,
            random_state=random_state,
            subpopulation_accuracies=test_val_subpopulation_accuracies,
            subpopulation_ratios=initial_subpopulation_ratios,
            base_size=(
                args.base_size * (args.split_ratios[key] / args.split_ratios["train"])
                if key != "test"
                else args.test_base_size
            ),
            is_train_set=(key == "train"),
            full_covariate_shift=args.full_covariate_shift,
        )
    print("train_dataset_size: ", len(datasets["train"]))
    print("val_dataset_size: ", len(datasets["val"]))
    print("test_dataset_size: ", len(datasets["test"]))
    return (
        datasets["train"],
        datasets["val"],
        datasets["test"],
        sampled_subpopulation_indices,
    )


def performative_shift(
    concatenated_datasets: ConcatDatasetWithDomainLabel,
    random_state: np.random.RandomState,
    shift_type: str = "domain",
    subpopulation_ratios: Dict[str, float] = None,
    subpopulation_accuracies: Dict = None,
    temperature: float = 1,
    positive_correlation: bool = False,
    base_size: int = -1,
    is_train_set: bool = False,
    full_covariate_shift: bool = False,
) -> List[Subset]:

    assert shift_type in ["domain", "class", "domain_class"]
    # get the subpopulation ratios
    if subpopulation_accuracies:
        assert subpopulation_ratios is None
        subpopulation_ratios = get_subpopulation_ratios(
            subpopulation_accuracies=subpopulation_accuracies,
            temperature=temperature,
            positive_correlation=positive_correlation,
        )

    print(
        f"We have a subpopulation with max {max(subpopulation_ratios.items(), key=lambda x: x[1])} and min {min(subpopulation_ratios.items(),key=lambda x: x[1])}."
    )

    # adjust the subpopulation dataset
    sampled_subpopulation_indices = partition_subpopulation_shift_dataset(
        dataset=concatenated_datasets,
        subpopulation_ratios=subpopulation_ratios,
        random_state=random_state,
        shift_type=shift_type,
        base_size=base_size,
        is_train_set=is_train_set,
        full_covariate_shift=full_covariate_shift,
    )

    adjusted_index_set = functools.reduce(
        lambda a, b: a + b, sampled_subpopulation_indices.values()
    )
    return (
        SubsetDatasetWithSampleGroup(
            dataset=concatenated_datasets,
            indices=adjusted_index_set,
            subpopulation_indices=sampled_subpopulation_indices,
        ),
        sampled_subpopulation_indices,
    )


def stage_path(data_dir, name):
    full_path = os.path.join(data_dir, name)

    if not os.path.exists(full_path):
        os.makedirs(full_path)

    return full_path


def download_and_extract(url, dst, remove=True):
    gdown.download(url, dst, quiet=False)

    if dst.endswith(".tar.gz"):
        tar = tarfile.open(dst, "r:gz")
        tar.extractall(os.path.dirname(dst))
        tar.close()

    if dst.endswith(".tar"):
        tar = tarfile.open(dst, "r:")
        tar.extractall(os.path.dirname(dst))
        tar.close()

    if dst.endswith(".zip"):
        zf = ZipFile(dst, "r")
        zf.extractall(os.path.dirname(dst))
        zf.close()

    if remove:
        os.remove(dst)


def download_pacs(data_dir):
    # Original URL: http://www.eecs.qmul.ac.uk/~dl307/project_iccv2017
    full_path = stage_path(data_dir, "PACS")

    download_and_extract(
        "https://drive.google.com/uc?id=1JFr8f805nMUelQWWmfnJR3y4_SYoN5Pd",
        os.path.join(data_dir, "PACS.zip"),
    )

    os.rename(os.path.join(data_dir, "kfold"), full_path)


def download_terra(data_dir):
    # Original URL: https://beerys.github.io/CaltechCameraTraps/
    # New URL: http://lila.science/datasets/caltech-camera-traps
    full_path = stage_path(data_dir, "TerraIncognita")
    download_and_extract(
        "https://lilablobssc.blob.core.windows.net/caltechcameratraps/eccv_18_all_images_sm.tar.gz",
        os.path.join(full_path, "terra_incognita_images.tar.gz"),
    )
    download_and_extract(
        "https://lilablobssc.blob.core.windows.net/caltechcameratraps/labels/caltech_camera_traps.json.zip",
        os.path.join(full_path, "caltech_camera_traps.json.zip"),
    )
    include_locations = ["38", "46", "100", "43"]
    include_categories = [
        "bird",
        "bobcat",
        "cat",
        "coyote",
        "dog",
        "empty",
        "opossum",
        "rabbit",
        "raccoon",
        "squirrel",
    ]
    images_folder = os.path.join(full_path, "eccv_18_all_images_sm/")
    annotations_file = os.path.join(full_path, "caltech_images_20210113.json")
    destination_folder = full_path
    stats = {}
    if not os.path.exists(destination_folder):
        os.mkdir(destination_folder)

    with open(annotations_file, "r") as f:
        data = json.load(f)

    category_dict = {}
    for item in data["categories"]:
        category_dict[item["id"]] = item["name"]

    for image in data["images"]:
        image_location = image["location"]
        if image_location not in include_locations:
            continue
        loc_folder = os.path.join(
            destination_folder, "location_" + str(image_location) + "/"
        )
        if not os.path.exists(loc_folder):
            os.mkdir(loc_folder)
        image_id = image["id"]
        image_fname = image["file_name"]
        for annotation in data["annotations"]:
            if annotation["image_id"] == image_id:
                if image_location not in stats:
                    stats[image_location] = {}
                category = category_dict[annotation["category_id"]]
                if category not in include_categories:
                    continue
                if category not in stats[image_location]:
                    stats[image_location][category] = 0
                else:
                    stats[image_location][category] += 1
                loc_cat_folder = os.path.join(loc_folder, category + "/")
                if not os.path.exists(loc_cat_folder):
                    os.mkdir(loc_cat_folder)
                dst_path = os.path.join(loc_cat_folder, image_fname)
                src_path = os.path.join(images_folder, image_fname)
                shutil.copyfile(src_path, dst_path)
    shutil.rmtree(images_folder)
    os.remove(annotations_file)


def create_image_list(data_dir, dataset_name):
    if dataset_name == "DollarStreet":
        dir_name = os.path.join(data_dir, "image_list")
        if os.path.isdir(dir_name) and ("image_list" in os.listdir(data_dir)):
            return
        root = data_dir
        train_pth = "images_v2_imagenet_train.csv"
        test_pth = "images_v2_imagenet_test.csv"

        df_train = pd.read_csv(os.path.join(root, train_pth))
        df_test = pd.read_csv(os.path.join(root, test_pth))

        df = pd.concat([df_train, df_test])
        regions = np.array(df["region.id"])
        dct = {}
        for r in regions:
            if r not in dct:
                dct[r] = 1
            else:
                dct[r] += 1
        image_pth = "assets/"
        image_dir = os.path.join(root, image_pth)
        image_list = os.path.join(root, "image_list")
        os.makedirs(image_list, exist_ok=True)
        num = 0
        imagenet_to_num = {}
        for rg in dct.keys():
            print(rg)
            f_pth = os.path.join(image_list, rg + "_all.txt")
            sub_df = df[df["region.id"] == rg]
            paths = np.array(sub_df["imageRelPath"])
            labels = np.array(sub_df["imagenet_sysnet_id"])
            with open(f_pth, "w") as file:
                for i in range(len(paths)):
                    str_label = np.array(ast.literal_eval(labels[i]))[0]
                    if str_label not in imagenet_to_num:
                        imagenet_to_num[str_label] = num
                        num += 1
                    txt = paths[i] + " " + str(imagenet_to_num[str_label]) + "\n"
                    file.write(txt)
    else:
        dir_name = os.path.join(data_dir, "image_list")
        if os.path.isdir(dir_name) and ("image_list" in os.listdir(data_dir)):
            return
        domain_list = os.listdir(data_dir)
        os.makedirs(dir_name, exist_ok=True)
        for dom in domain_list:
            dom_path = os.path.join(dir_name, dom + "_all.txt")
            with open(dom_path, "w") as file:
                read_domain = os.path.join(data_dir, dom)
                for i, label in enumerate(sorted(os.listdir(read_domain))):
                    read_class_dom = os.path.join(read_domain, label)
                    for img_pth in os.listdir(read_class_dom):
                        txt = os.path.join(dom, label, img_pth) + " " + str(i) + "\n"
                        file.write(txt)


def get_subpopulation_shift_dataset(
    dataset_name,
    root,
    split_ratios: Dict[str, float],
    download=True,
    train_transform=None,
    val_transform=None,
    seed=0,
):
    assert set(split_ratios.keys()) == {"train", "val", "test"}
    # load datasets from tllib.vision.datasets
    supported_dataset = [
        "PACS",
        "OfficeHome",
        "DomainNet",
        "TerraIncognita",
        "DollarStreet",
        "CivilComments",
        "Amazon",
        "CIFAR10",
        "CIFAR100",
        "ImageNet100",
        "AGNews",
    ] + TABULAR_DATASET_NAMES
    assert dataset_name in supported_dataset
    train_split_list = []
    val_split_list = []
    test_split_list = []

    if dataset_name in TABULAR_DATASET_NAMES:
        dataset = datasets.__dict__["TabularHFDataset"]
        domains = ["none"]
    else:
        dataset = datasets.__dict__[dataset_name]
        domains = dataset.domains()
    for domain_index, task in enumerate(domains):
        if dataset_name == "PACS":
            try:
                all_split = dataset(
                    root=root, task=task, split="all", download=download
                )
                num_classes = all_split.num_classes
            except:
                print("Resolving...")
                # download
                if not os.path.isdir(root) or len(os.listdir(root)) == 0:
                    download_pacs(root[: root.rfind("/")])

                create_image_list(root, dataset_name)
                all_split = dataset(
                    root=root, task=task, split="all", download=download
                )
                num_classes = all_split.num_classes

        elif dataset_name == "OfficeHome":
            all_split = dataset(root=root, task=task, download=download)
            num_classes = all_split.num_classes

        elif dataset_name == "DomainNet":
            train_split = dataset(
                root=root, task=task, split="train", download=download
            )
            test_split = dataset(root=root, task=task, split="test", download=download)

            train_split.samples += test_split.samples
            train_split.targets += test_split.targets
            all_split = train_split
            num_classes = all_split.num_classes

        elif dataset_name == "TerraIncognita":
            try:
                all_split = dataset(
                    root=root, task=task, split="all", download=download
                )
                num_classes = all_split.num_classes
            except:
                print("Resolving...")
                # download
                if not os.path.isdir(root) or len(os.listdir(root)) == 0:
                    download_terra(root[: root.rfind("/")])
                create_image_list(root, dataset_name)
                all_split = dataset(root=root, task=task, split="all", download=True)
                num_classes = all_split.num_classes
        elif dataset_name == "DollarStreet":
            """
            download the dataset from https://www.kaggle.com/datasets/mlcommons/the-dollar-street-dataset/data
            put it on the data directory with name DollarStreet
            """
            try:
                all_split = dataset(
                    root=root,
                    task=task,
                    split="all",
                    download=download,
                    init_preprocessing=False,
                    use_preprocessed_data=True,
                )
                num_classes = all_split.num_classes
            except:
                print("Resolving...")
                create_image_list(root, dataset_name)
                all_split = dataset(root=root, task=task, split="all", download=True)
                num_classes = all_split.num_classes
        elif dataset_name == "CivilComments":
            all_split = dataset(download=False, root_dir=root)
            num_classes = 2
            all_split.CLASSES = ["positive", "negative"]

            # mask out the bad samples (235182, 245642)
            mask = torch.ones(len(all_split), dtype=torch.bool)
            mask[235182] = False
            mask[245642] = False
            all_split._y_array = all_split._y_array[mask]
            del all_split._text_array[235182]  # since we deleted 235182th
            del all_split._text_array[
                245641
            ]  # to delete 245642th, decrement index by one

        elif dataset_name == "Amazon":
            all_split = dataset(download=False, root_dir=root)
            num_classes = 5
            all_split.CLASSES = ["1", "2", "3", "4", "5"]

        elif dataset_name in ["CIFAR10", "CIFAR100"]:
            train_split = dataset(root=root, split="train", download=download)
            test_split = dataset(root=root, split="test", download=download)

            train_split.data = np.concatenate(
                (train_split.data, test_split.data), axis=0
            )
            train_split.targets += test_split.targets
            all_split = train_split
            num_classes = all_split.num_classes

        elif dataset_name == "ImageNet100":
            all_split = dataset(root=os.path.join(root, "train"))
            num_classes = 100

        elif dataset_name == "AGNews":
            all_split = dataset()
            num_classes = 4

        elif dataset_name in TABULAR_DATASET_NAMES:
            all_split = dataset(hf_dataset_name=dataset_name)
            num_classes = all_split.num_classes

        else:
            raise NotImplementedError

        train_split, val_split, test_split = split_dataset_multiple(
            all_split,
            [
                int(len(all_split) * split_ratios["train"]),
                int(len(all_split) * split_ratios["val"]),
            ],
            seed,
        )
        train_split_list.append(train_split)
        val_split_list.append(val_split)
        test_split_list.append(test_split)

    train_dataset = ConcatDatasetWithDomainLabel(
        train_split_list, transform=train_transform
    )
    val_dataset = ConcatDatasetWithDomainLabel(val_split_list, transform=val_transform)
    test_dataset = ConcatDatasetWithDomainLabel(
        test_split_list, transform=val_transform
    )

    dataset_dict = {"train": train_dataset, "val": val_dataset, "test": test_dataset}

    return dataset_dict["train"], dataset_dict["val"], dataset_dict["test"], num_classes


def get_subpopulation_indices(
    stratified_category: str,
    concatenated_dataset: ConcatDatasetWithDomainLabel,
    full_covariate_shift: bool = False,
    domain_ids: Optional[int] = None,
):
    class_labels = []
    for dataset in concatenated_dataset.datasets:
        if isinstance(dataset.dataset, WILDSSubset):
            class_labels.append(
                dataset.dataset.dataset._y_array.numpy()[dataset.indices]
            )
        else:
            class_labels.append(np.array(dataset.dataset.targets)[dataset.indices])
    class_labels = np.concatenate(class_labels)
    domain_labels = np.array(list(concatenated_dataset.index_to_domain_id.values()))

    assert len(class_labels) == len(domain_labels)
    assert min(class_labels) == min(domain_labels) == 0

    num_classes = max(class_labels) + 1
    num_domains = max(domain_labels) + 1

    index_subpopulations = OrderedDict()

    if stratified_category == "domain_class":
        for domain_label in range(num_domains):
            for class_label in range(num_classes):
                domain_indices = np.where(domain_labels == domain_label)[0]
                class_indices = np.where(class_labels == class_label)[0]
                subpopulation_indices = np.intersect1d(
                    domain_indices, class_indices
                ).tolist()
                index_subpopulations[f"domain_{domain_label}_class_{class_label}"] = (
                    subpopulation_indices
                )
    elif stratified_category == "domain":
        for domain_label in range(num_domains):
            domain_indices = np.where(domain_labels == domain_label)[0].tolist()
            index_subpopulations[f"domain_{domain_label}"] = domain_indices
    elif stratified_category == "class":
        for class_label in range(num_classes):
            if full_covariate_shift:
                class_indices_with_domain = np.where(
                    (class_labels == class_label)
                    & (
                        (domain_labels == domain_ids[0])
                        | (domain_labels == domain_ids[1])
                    )
                )[0].tolist()
                index_subpopulations[f"class_{class_label}"] = class_indices_with_domain
            else:
                class_indices = np.where(class_labels == class_label)[0].tolist()
                index_subpopulations[f"class_{class_label}"] = class_indices
    else:
        raise NotImplementedError

    return index_subpopulations


def split_dataset_multiple(dataset, sizes, seed=0):
    assert sum(sizes) < len(dataset)
    idxes = list(range(len(dataset)))
    np.random.RandomState(seed).shuffle(idxes)
    subsets = []
    left_index = 0
    for size in sizes:
        subsets.append(Subset(dataset, idxes[left_index : left_index + size]))
        left_index += size
    subsets.append(Subset(dataset, idxes[left_index:]))
    return subsets


def str2bool(val):
    """Convert a string representation of truth to true (1) or false (0).
    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
    'val' is anything else.
    """
    val = val.lower()
    if val in ("y", "yes", "t", "true", "on", "1"):
        return 1
    elif val in ("n", "no", "f", "false", "off", "0"):
        return 0
    else:
        raise ValueError("invalid truth value %r" % (val,))


