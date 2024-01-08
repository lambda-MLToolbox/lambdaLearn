from lambdaLearn.Dataset.CILDataset import CILDataset, DummyDataset
from lambdaLearn.Base.VisionMixin import VisionMixin

from torchvision import datasets, transforms
import numpy as np


class CIFAR100_CIL(CILDataset, VisionMixin):
    base_folder = "cifar-100-python"
    class_order = np.arange(100).tolist()

    def __init__(
            self,
            root='./data/',
            shuffle=False,
            random_state=None,
            download: bool = False,
    ):
        super(CIFAR100_CIL, self).__init__(dataset_name='CIFAR100', init_cls=10, increment=10)
        self.train_transform = None
        self.root = root
        self.shuffle = shuffle
        self.random_state = random_state
        self.download = download

        self.use_path = False

        self._train_trsf = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=63 / 255),
            transforms.ToTensor()
        ]
        self._test_trsf = [transforms.ToTensor()]
        self._common_trsf = [
            transforms.Normalize(
                mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)
            ),
        ]

        self.nb_tasks = 10  # TODO

        self.init_dataset()

    def _download_data(self):
        train_dataset = datasets.cifar.CIFAR100(self.root, train=True, download=self.download)
        test_dataset = datasets.cifar.CIFAR100(self.root, train=False, download=self.download)
        self._train_data, self._train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self._test_data, self._test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )

    def init_dataset(self, images=None, labels=None) -> None:
        self._download_data()

        self._update_order()


