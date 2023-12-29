from abc import ABC, abstractmethod

from sklearn.base import BaseEstimator, TransformerMixin


class Transformer(BaseEstimator, TransformerMixin, ABC):
    def __init__(self):
        pass

    def fit(self, X, y=None, **fit_params):
        """
        Obtain the processing function through existing data.

        :param X: Samples for learning the function of transformation.
        :param y: Labels for learning the function of transformation.
        """
        return self

    def __call__(self, X, y=None, **fit_params):
        """
        Same as fit_transform.

        :param X: Samples for learning and transformation.
        :param y: labels for learning.
        """
        return self.fit_transform(X, y, fit_params=fit_params)

    @abstractmethod
    def transform(self, X):
        """
        Process the new data.

        :param X: Data to be converted.
        """
        raise NotImplementedError("Transform method of Augmentation class must be implemented.")

    def fit_transform(self, X, y=None, **fit_params):
        """
        Firstly perform fit() on the existing samples X and labels y, and then directly transform y.

        :param X: Samples for learning and transformation.
        :param y: Labels fo learning
        """
        return self.fit(X=X, y=y, fit_params=fit_params).transform(X)
