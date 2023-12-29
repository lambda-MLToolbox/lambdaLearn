import numpy as np
import PIL
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode

from lambdaLearn.Base.Transformer import Transformer
from lambdaLearn.utils import partial


class Resize(Transformer):
    def __init__(
        self,
        size,
        interpolation=InterpolationMode.BILINEAR,
        max_size=None,
        antialias=None,
    ):
        """
        :param size: Desired output size. If size is a sequence like (h, w), the output size
            will be matched to this. If size is an int, the smaller edge of the image
            will be matched to this number maintaining the aspect ratio.
        :param interpolation: Desired interpolation enum defined by 'torchvision.transforms.InterpolationMode'.
        :param max_size: The maximum allowed for the longer edge of the resized image: if the
            longer edge of the image is greater than 'max_size`' after being resized
            according to 'size', then the image is resized again so that the longer
            edge is equal to 'max_size'.
        :param antialias: antialias flag. If 'img' is PIL Image, the flag is ignored and anti-alias
            is always used. If 'img' is Tensor, the flag is False by default and can be set
            to True for 'InterpolationMode.BILINEAR' only mode. This can help making the output
            for PIL images and tensors closer.
        """

        super().__init__()
        self.resize = partial(
            F.resize,
            size=size,
            interpolation=interpolation,
            max_size=max_size,
            antialias=antialias,
        )

    def transform(self, X):
        if isinstance(X, np.ndarray):
            X = PIL.Image.fromarray(X)
        X = self.resize(X)
        return X
