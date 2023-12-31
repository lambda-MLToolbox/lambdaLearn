import numpy as np
import PIL
import torch
import torchvision.transforms.functional as F

from lambdaLearn.Base.Transformer import Transformer


class Brightness(Transformer):
    def __init__(self, min_v=0.05, max_v=0.95, num_bins=10, magnitude=5, v=None):
        super().__init__()
        """
        :param min_v: The minimum value of the augmentation.
        :param max_v: The maximum value of the augmentation.
        :param num_bins: The number of intervals  division for the value of the augmentation.
        :param magnitude: The level of the augmentation.
        :param v: Specify the value of the augmentation directly.
        """
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
            # _v = self.v if random.random() < 0.5 else self.v * -1
            _v = self.v
            X = PIL.ImageEnhance.Brightness(X).enhance(_v)
            return X
        elif isinstance(X, torch.Tensor):
            if len(X.shape) == 4:
                for _ in range(X.shape[0]):
                    # _v = self.v if random.random() < 0.5 else self.v * -1
                    _v = self.v
                    X[_] = F.adjust_brightness(X[_], _v)
            else:
                # _v = self.v if random.random() < 0.5 else self.v * -1
                _v = self.v
                X = F.adjust_brightness(X, _v)
            return X
        else:
            raise ValueError("No data to augment")
