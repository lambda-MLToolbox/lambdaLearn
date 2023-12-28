import os
import pickle

import numpy as np
from torchvision.datasets.utils import check_integrity, download_and_extract_archive

from lambdaLearn.Base.VisionMixin import VisionMixin
from lambdaLearn.Dataset.LabeledDataset import LabeledDataset
from lambdaLearn.Dataset.SemiDataset import SemiDataset
from lambdaLearn.Dataset.TrainDataset import TrainDataset
from lambdaLearn.Dataset.UnlabeledDataset import UnlabeledDataset
from lambdaLearn.Split.DataSplit import DataSplit


class CIFAR100(SemiDataset, VisionMixin):
    base_folder = "cifar-100-python"
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = "eb9058c3a382ffc7106e4002c42a8d85"
    train_list = [
        ["train", "16019d7e3df5f24257cddd939b257f8d"],
    ]

    test_list = [
        ["test", "f0ef6b0ae62326f3e7ffdfab6717acfc"],
    ]
    meta = {
        "filename": "meta",
        "key": "fine_label_names",
        "md5": "7973b15100ade9c7d40fb424638fde48",
    }
    mean = [0.5071, 0.4865, 0.4409]  # [0.4914, 0.4822, 0.4465]
    std = [0.2673, 0.2564, 0.2762]  # [0.2471, 0.2435, 0.2616]

    def __init__(
        self,
        root: str,
        default_transforms=False,
        pre_transform=None,
        transforms=None,
        transform=None,
        target_transform=None,
        unlabeled_transform=None,
        valid_transform=None,
        test_transform=None,
        valid_size=None,
        labeled_size=0.1,
        stratified=False,
        shuffle=True,
        random_state=None,
        download: bool = False,
    ) -> None:
        self.default_transforms = default_transforms
        self.labeled_X = None
        self.labeled_y = None
        self.unlabeled_X = None
        self.unlabeled_y = None
        self.valid_X = None
        self.valid_y = None
        self.test_X = None
        self.test_y = None

        self.labeled_dataset = None
        self.unlabeled_dataset = None
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None

        self.data_initialized = False

        self.len_test = None
        self.len_valid = None
        self.len_labeled = None
        self.len_unlabeled = None

        self.labeled_X_indexing_method = None
        self.labeled_y_indexing_method = None
        self.unlabeled_X_indexing_method = None
        self.unlabeled_y_indexing_method = None
        self.valid_X_indexing_method = None
        self.valid_indexing_method = None
        self.test_X_indexing_method = None
        self.test_y_indexing_method = None

        SemiDataset.__init__(
            self,
            pre_transform=pre_transform,
            transforms=transforms,
            transform=transform,
            target_transform=target_transform,
            unlabeled_transform=unlabeled_transform,
            test_transform=test_transform,
            valid_transform=valid_transform,
            labeled_size=labeled_size,
            valid_size=valid_size,
            stratified=stratified,
            shuffle=shuffle,
            random_state=random_state,
        )
        VisionMixin.__init__(self, mean=self.mean, std=self.std)

        if isinstance(root, (str, bytes)):
            root = os.path.expanduser(root)
        self.root = root

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        self._load_meta()
        if self.default_transforms:
            self.init_default_transforms()
        self.init_dataset()

    def _load_meta(self) -> None:
        path = os.path.join(self.root, self.base_folder, self.meta["filename"])
        if not check_integrity(path, self.meta["md5"]):
            raise RuntimeError("Dataset metadata file not found or corrupted. You can use download=True to download it")
        with open(path, "rb") as infile:
            data = pickle.load(infile, encoding="latin1")
            self.classes = data[self.meta["key"]]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def _check_integrity(self) -> bool:
        root = self.root
        for fentry in self.train_list + self.test_list:
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self) -> None:
        if self._check_integrity():
            print("Files already downloaded and verified")
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def _init_dataset(self):
        test_X = []
        test_y = []
        for file_name, checksum in self.test_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                test_X.append(entry["data"])
                if "labels" in entry:
                    test_y.extend(entry["labels"])
                else:
                    test_y.extend(entry["fine_labels"])
        test_X = np.vstack(test_X).reshape(-1, 3, 32, 32)
        test_X = test_X.transpose((0, 2, 3, 1))

        train_X = []
        train_y = []
        for file_name, checksum in self.train_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                train_X.append(entry["data"])
                if "labels" in entry:
                    train_y.extend(entry["labels"])
                else:
                    train_y.extend(entry["fine_labels"])
        train_X = np.vstack(train_X).reshape(-1, 3, 32, 32)
        train_X = train_X.transpose((0, 2, 3, 1))

        if self.valid_size is not None:
            valid_X, valid_y, train_X, train_y = DataSplit(
                X=train_X,
                y=train_y,
                size_split=self.valid_size,
                stratified=self.stratified,
                shuffle=self.shuffle,
                random_state=self.random_state,
            )
        else:
            valid_X = None
            valid_y = None

        if self.labeled_size is not None:
            labeled_X, labeled_y, unlabeled_X, unlabeled_y = DataSplit(
                X=train_X,
                y=train_y,
                size_split=self.labeled_size,
                stratified=self.stratified,
                shuffle=self.shuffle,
                random_state=self.random_state,
            )
        else:
            labeled_X, labeled_y = train_X, train_y
            unlabeled_X, unlabeled_y = None, None
        self.test_dataset = LabeledDataset(pre_transform=self.pre_transform, transform=self.test_transform)
        self.test_dataset.init_dataset(test_X, test_y)
        self.valid_dataset = LabeledDataset(pre_transform=self.pre_transform, transform=self.valid_transform)
        self.valid_dataset.init_dataset(valid_X, valid_y)
        self.train_dataset = TrainDataset(
            pre_transform=self.pre_transform,
            transforms=self.transforms,
            transform=self.transform,
            target_transform=self.target_transform,
            unlabeled_transform=self.unlabeled_transform,
        )
        labeled_dataset = LabeledDataset(
            pre_transform=self.pre_transform,
            transforms=self.transforms,
            transform=self.transform,
            target_transform=self.target_transform,
        )
        labeled_dataset.init_dataset(labeled_X, labeled_y)
        unlabeled_dataset = UnlabeledDataset(pre_transform=self.pre_transform, transform=self.unlabeled_transform)
        unlabeled_dataset.init_dataset(unlabeled_X, unlabeled_y)
        self.train_dataset.init_dataset(labeled_dataset=labeled_dataset, unlabeled_dataset=unlabeled_dataset)
