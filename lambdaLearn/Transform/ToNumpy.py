import numpy as np

from lambdaLearn.Base.Transformer import Transformer
from lambdaLearn.utils import is_pandas_ndframe, is_torch_data_type, to_numpy


class ToNumpy(Transformer):
    def __init__(self):
        super().__init__()

    def transform(self, X):
        if isinstance(X, np.ndarray):
            return X
        if isinstance(X, dict):
            return np.array(to_numpy(val) for key, val in X.items())
        if is_pandas_ndframe(X):
            return X.values
        if isinstance(X, (tuple, list)):
            return np.array(X)
        if not is_torch_data_type(X):
            raise TypeError("Cannot convert this data type to a numpy array.")
        if X.is_cuda:
            X = X.cpu()
        if X.requires_grad:
            X = X.detach()
        return X.numpy()
