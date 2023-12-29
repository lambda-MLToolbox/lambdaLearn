import numpy as np
import PIL.Image
import torch
import torchvision.transforms.functional as F

from lambdaLearn.Base.Transformer import Transformer


class Solarize(Transformer):
    def __init__(self, min_v=0, max_v=255, num_bins=10, magnitude=5, v=None, scale=256):
        """
        :param min_v: The minimum value of the augmentation.
        :param max_v: The maximum value of the augmentation.
        :param num_bins: The number of intervals  division for the value of the augmentation.
        :param magnitude: The level of the augmentation.
        :param v: Specify the value of the augmentation directly.
        :param scale: Scale of image pixel values
        """
        super().__init__()
        self.max_v = max_v
        self.min_v = min_v
        self.num_bins = num_bins
        self.magnitude = magnitude
        self.magnitudes = torch.linspace(self.max_v, self.min_v, self.num_bins)
        self.scale = scale
        self.v = float(self.magnitudes[self.magnitude - 1].item()) if v is None else v

    def transform(self, X):
        if isinstance(X, np.ndarray):
            X = PIL.Image.fromarray(X)
        if isinstance(X, PIL.Image.Image):
            X = PIL.ImageOps.solarize(X, self.scale - self.v)
            return X
        elif isinstance(X, torch.Tensor):
            X = F.solarize((X * (self.scale - 1)).ceil().type(torch.uint8), self.scale - self.v) / self.scale
            return X
        else:
            raise ValueError("No data to augment")
