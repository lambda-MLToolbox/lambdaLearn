import numpy as np
import PIL
import torch
from PIL import Image

from lambdaLearn.Base.Transformer import Transformer


class ToImage(Transformer):
    def __init__(self, channels=3, channels_first=False, load_from_path=False, format=None):
        # > - Parameter:
        # >> - channels: The number of channels of input images.
        # >> - channels_first: Whether the number of channels is before the image size.
        super(ToImage, self).__init__()
        self.channels = channels
        self.channels_first = channels_first
        self.load_from_path = load_from_path
        self.format = format

    def transform(self, X):
        if self.load_from_path:
            X = Image.open(X).convert(self.format)
            return X
        if isinstance(X, torch.Tensor):
            X = X.numpy()
        if self.channels_first is True and len(X.shape) == 3:
            X = X.transpose((1, 2, 0))
        if isinstance(X, np.ndarray):
            X = PIL.Image.fromarray(X)
        return X
