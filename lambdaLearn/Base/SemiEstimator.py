from abc import ABC, abstractmethod

from sklearn.base import BaseEstimator


class SemiEstimator(ABC, BaseEstimator):
    @abstractmethod
    def fit(self, X, y, unlabeled_X):
        """
        Train a SSL model.

        :param X: Instances of labeled data.
        :param y: Labels of labeled data.
        :param unlabeled_X: Instances of unlabeled data.
        """
        raise NotImplementedError("The fit() method of SemiEstimator must be implemented.")
