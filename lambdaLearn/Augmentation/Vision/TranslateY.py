import random

import numpy as np
import PIL
import torch
import torchvision.transforms.functional as F

from lambdaLearn.Base.Transformer import Transformer


class TranslateY(Transformer):
    def __init__(self, min_v=0, max_v=0.3, num_bins=10, magnitude=5, v=None):
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
        self.magnitudes = torch.linspace(min_v, max_v, num_bins)
        self.v = float(self.magnitudes[self.magnitude].item()) if v is None else v

    def transform(self, X):
        if isinstance(X, np.ndarray):
            X = PIL.Image.fromarray(X)
        if isinstance(X, PIL.Image.Image):
            _v = self.v if random.random() < 0.5 else self.v * -1
            _v = int(_v * X.size[1])
            X = X.transform(X.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, _v))
            return X
        elif isinstance(X, torch.Tensor):
            if len(X.shape) == 4:
                for _ in range(X.shape[0]):
                    _v = self.v if random.random() < 0.5 else self.v * -1
                    _v = int(_v * X.shape[-1])
                    X[_] = F.affine(X[_], angle=0, translate=[0, _v], scale=1.0, shear=[0.0, 0.0])
            else:
                _v = self.v if random.random() < 0.5 else self.v * -1
                _v = int(_v * X.shape[-1])
                X = F.affine(X, angle=0, translate=[0, _v], scale=1.0, shear=[0.0, 0.0])
            return X
        else:
            raise ValueError("No data to augment")
