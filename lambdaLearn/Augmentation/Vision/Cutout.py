import numpy as np
import PIL
import PIL.ImageEnhance
import torch

from lambdaLearn.Augmentation.Vision.CutoutAbs import CutoutAbs
from lambdaLearn.Base.Transformer import Transformer


class Cutout(Transformer):
    def __init__(self, v=0.5, fill=(127, 127, 127), random_v=True):
        """
        :param v: The relative value of crop size.
        :param fill: The padding value.
        :param random_v: Whether to randomly determine the crop size.
        """
        super().__init__()
        self.v = v
        self.fill = fill
        self.random_v = random_v
        assert 0.0 <= v <= 0.5

    def transform(self, X):
        if isinstance(X, np.ndarray):
            X = PIL.Image.fromarray(X)
        if isinstance(X, PIL.Image.Image):
            v = self.v * X.size[0]
        elif isinstance(X, torch.Tensor):
            v = self.v * X.shape[-2]
        else:
            raise ValueError("No data to augment")
        X = CutoutAbs(v, self.fill, self.random_v).fit_transform(X)

        return X
