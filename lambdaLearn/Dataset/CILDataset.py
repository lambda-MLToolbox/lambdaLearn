import copy
from PIL import Image
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset


class CILDataset(Dataset):
    def __init__(self, dataset_name, init_cls, increment, shuffle=False, random_state=None):
        self.images = None
        self.labels = None
        self.trsf = None
        self.use_path = False
        self.shuffle = shuffle
        self.random_state = random_state

        assert self.random_state is not None if self.shuffle else True

        self.dataset_name = dataset_name

        self._class_order = None
        self._init_cls = init_cls
        self.increment = increment
        self.nb_tasks = 0

        self._train_data = None
        self._train_targets = None
        self._test_data = None
        self._test_targets = None

        # Default transforms
        self._train_trsf = [transforms.ToTensor()]
        self._test_trsf = [transforms.ToTensor()]
        self._common_trsf = []

        self._increments = [init_cls]

    def init_dataset(self, images=None, labels=None):
        raise NotImplementedError

    def __getitem__(self, idx, test=False, valid=False):
        if test:
            return idx, self._test_data[idx], self._test_targets[idx]

        return idx, self._train_data[idx], self._train_targets[idx]

    def __len__(self, test=False, valid=False):
        if test:
            return len(self._test_data)

        return len(self._train_data)

    def get_dataset(self, indices, source, mode, appendent=None, ret_data=False, m_rate=None):
        """
        Acquire the data whose labels are specified .
        """
        if source == "train":
            x, y = self._train_data, self._train_targets
        elif source == "test":
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError("Unknown data source {}.".format(source))

        if mode == "train":
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == "flip":
            trsf = transforms.Compose(
                [
                    *self._test_trsf,
                    transforms.RandomHorizontalFlip(p=1.0),
                    *self._common_trsf,
                ]
            )
        elif mode == "test":
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError("Unknown mode {}.".format(mode))

        data, targets = [], []
        for idx in indices:
            if m_rate is None:
                class_data, class_targets = self._select(
                    x, y, low_range=idx, high_range=idx + 1
                )
            else:
                class_data, class_targets = self._select_rmm(
                    x, y, low_range=idx, high_range=idx + 1, m_rate=m_rate
                )
            data.append(class_data)
            targets.append(class_targets)

        if appendent is not None and len(appendent) != 0:
            appendent_data, appendent_targets = appendent
            data.append(appendent_data)
            targets.append(appendent_targets)

        data, targets = np.concatenate(data), np.concatenate(targets)

        if ret_data:
            return data, targets, DummyDataset(data, targets, trsf, self.use_path)
        else:
            return DummyDataset(data, targets, trsf, self.use_path)

    def get_finetune_dataset(self, known_classes, total_classes, source, mode, appendent, type="ratio"):
        if source == 'train':
            x, y = self._train_data, self._train_targets
        elif source == 'test':
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError('Unknown data source {}.'.format(source))

        if mode == 'train':
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == 'test':
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError('Unknown mode {}.'.format(mode))
        val_data = []
        val_targets = []

        old_num_tot = 0
        appendent_data, appendent_targets = appendent

        for idx in range(0, known_classes):
            append_data, append_targets = self._select(appendent_data, appendent_targets,
                                                       low_range=idx, high_range=idx + 1)
            num = len(append_data)
            if num == 0:
                continue
            old_num_tot += num
            val_data.append(append_data)
            val_targets.append(append_targets)
        if type == "ratio":
            new_num_tot = int(old_num_tot * (total_classes - known_classes) / known_classes)
        elif type == "same":
            new_num_tot = old_num_tot
        else:
            assert 0, "not implemented yet"
        new_num_average = int(new_num_tot / (total_classes - known_classes))
        for idx in range(known_classes, total_classes):
            class_data, class_targets = self._select(x, y, low_range=idx, high_range=idx + 1)
            val_indx = np.random.choice(len(class_data), new_num_average, replace=False)
            val_data.append(class_data[val_indx])
            val_targets.append(class_targets[val_indx])
        val_data = np.concatenate(val_data)
        val_targets = np.concatenate(val_targets)
        return DummyDataset(val_data, val_targets, trsf, self.use_path)

    def get_dataset_with_split(self, indices, source, mode, appendent=None, val_samples_per_class=0):
        if source == "train":
            x, y = self._train_data, self._train_targets
        elif source == "test":
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError("Unknown data source {}.".format(source))

        if mode == "train":
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == "test":
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError("Unknown mode {}.".format(mode))

        train_data, train_targets = [], []
        val_data, val_targets = [], []
        for idx in indices:
            class_data, class_targets = self._select(
                x, y, low_range=idx, high_range=idx + 1
            )
            val_indx = np.random.choice(
                len(class_data), val_samples_per_class, replace=False
            )
            train_indx = list(set(np.arange(len(class_data))) - set(val_indx))
            val_data.append(class_data[val_indx])
            val_targets.append(class_targets[val_indx])
            train_data.append(class_data[train_indx])
            train_targets.append(class_targets[train_indx])

        if appendent is not None:
            appendent_data, appendent_targets = appendent
            for idx in range(0, int(np.max(appendent_targets)) + 1):
                append_data, append_targets = self._select(
                    appendent_data, appendent_targets, low_range=idx, high_range=idx + 1
                )
                val_indx = np.random.choice(
                    len(append_data), val_samples_per_class, replace=False
                )
                train_indx = list(set(np.arange(len(append_data))) - set(val_indx))
                val_data.append(append_data[val_indx])
                val_targets.append(append_targets[val_indx])
                train_data.append(append_data[train_indx])
                train_targets.append(append_targets[train_indx])

        train_data, train_targets = np.concatenate(train_data), np.concatenate(
            train_targets
        )
        val_data, val_targets = np.concatenate(val_data), np.concatenate(val_targets)

        return DummyDataset(
            train_data, train_targets, trsf, self.use_path
        ), DummyDataset(val_data, val_targets, trsf, self.use_path)

    def _select(self, x, y, low_range, high_range):
        idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]

        if isinstance(x, np.ndarray):
            x_return = x[idxes]
        else:
            x_return = []
            for id in idxes:
                x_return.append(x[id])
        return x_return, y[idxes]

    def _update_order(self):
        order = [i for i in range(len(np.unique(self._train_targets)))]
        if self.shuffle:
            np.random.seed(self.random_state)
            order = np.random.permutation(len(order)).tolist()
        else:
            order = self.class_order

        self._class_order = order

        self._train_targets = self._map_new_class_index(self._train_targets, order)
        self._test_targets = self._map_new_class_index(self._test_targets, order)

        while sum(self._increments) + self.increment < len(self._class_order):
            self._increments.append(self.increment)
        offset = len(self._class_order) - sum(self._increments)
        if offset > 0:
            self._increments.append(offset)

    def get_task_size(self, task):
        return self._increments[task]

    def get_accumulate_tasksize(self,task) -> int:
        return sum(self._increments[:task+1])

    def getlen(self, index):
        y = self._train_targets
        return np.sum(np.where(y == index))

    @staticmethod
    def _map_new_class_index(y, order):
        return np.array(list(map(lambda x: order.index(x), y)))


class DummyDataset(Dataset):
    def __init__(self, images, labels, trsf, use_path=False):
        assert len(images) == len(labels), "Data size error!"
        self.images = images
        self.labels = labels
        self.trsf = trsf
        self.use_path = use_path

    def __getitem__(self, idx):
        if self.use_path:
            with open(self.images[idx], "rb") as f:
                img = Image.open(f)
                image = self.trsf(img.convert("RGB"))
        else:
            image = self.trsf(Image.fromarray(self.images[idx]))
        label = self.labels[idx]

        return idx, image, label

    def __len__(self):
        return len(self.images)
