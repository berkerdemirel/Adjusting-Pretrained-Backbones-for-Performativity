from typing import Optional
import os
from .imagelist import ImageList
from ._util import download as download_data, check_exits


class TerraIncognita(ImageList):
    """`TerraIncognita Dataset`_.

    Args:
        root (str): Root directory of dataset
        task (str): The task (domain) to create dataset. Choices include ``'A'``: amazon, \
            ``'D'``: dslr and ``'W'``: webcam.
        download (bool, optional): If true, downloads the dataset from the internet and puts it \
            in root directory. If dataset is already downloaded, it is not downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image and returns a \
            transformed version. E.g, :class:`torchvision.transforms.RandomCrop`.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.

    .. note:: In `root`, there will exist following files after downloading.
        ::
            38/
                bird/
                    *.jpg
                    ...
            46/
            100/
            43
            image_list/
                38.txt
                46.txt
                100.txt
                43.txt
    """
    download_list = [

    ]
    # "38", "46", "100", "43"
    image_list = {
        "38": "image_list/location_38_{}.txt",
        "46": "image_list/location_46_{}.txt",
        "100": "image_list/location_100_{}.txt",
        "43": "image_list/location_43_{}.txt"
    }
    CLASSES = ["bird", "bobcat", "cat", "coyote", "dog", "empty", "opossum", "rabbit", "raccoon", "squirrel"]

    def __init__(self, root: str, task: str, split='all', download: Optional[bool] = True, **kwargs):
        assert task in self.image_list
        assert split in ["train", "val", "all", "test"]
        if split == "test":
            split = "all"
        data_list_file = os.path.join(root, self.image_list[task].format(split))

        if download:
            list(map(lambda args: download_data(root, *args), self.download_list))
        else:
            list(map(lambda file_name, _: check_exits(root, file_name), self.download_list))

        super(TerraIncognita, self).__init__(root, TerraIncognita.CLASSES, data_list_file=data_list_file, **kwargs)

    @classmethod
    def domains(cls):
        return list(cls.image_list.keys())
