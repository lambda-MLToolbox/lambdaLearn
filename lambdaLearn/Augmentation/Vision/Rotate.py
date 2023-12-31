import random

import numpy as np
import PIL
import torch
import torchvision.transforms.functional as F

from lambdaLearn.Base.Transformer import Transformer


class Rotate(Transformer):
    def __init__(self, min_v=0, max_v=30, num_bins=10, magnitude=5, v=None):
        """
        :param min_v: The minimum value of the augmentation.
        :param max_v: The maximum value of the augmentation.
        :param num_bins: The number of intervals  division for the value of the augmentation.
        :param magnitude: The level of the augmentation.
        :param v: Specify the value of the augmentation directly.
        """
        super().__init__()
        self.max_v = max_v
        self.min_v = min_v
        self.num_bins = num_bins
        self.magnitude = magnitude
        if v is None:
            self.magnitudes = torch.linspace(min_v, max_v, num_bins)
            self.v = float(self.magnitudes[self.magnitude].item())
        else:
            self.v = v

    def transform(self, X, rand=False):
        if isinstance(X, np.ndarray):
            X = PIL.Image.fromarray(X)
        if isinstance(X, PIL.Image.Image):
            _v = self.v if random.random() < 0.5 else self.v * -1
            X = X.rotate(_v)
            return X
        elif isinstance(X, torch.Tensor):
            if len(X.shape) == 4:
                for _ in range(X.shape[0]):
                    _v = self.v if random.random() < 0.5 else self.v * -1
                    X[_] = F.rotate(X[_], _v)
            else:
                _v = self.v if random.random() < 0.5 else self.v * -1
                X = F.rotate(X, _v)
            return X

        else:
            raise ValueError("No data to augment")
