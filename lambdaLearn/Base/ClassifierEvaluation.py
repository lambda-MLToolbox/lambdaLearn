from abc import ABC, abstractmethod


class ClassifierEvaluation(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def scoring(self, y_true, y_pred=None, y_score=None):
        """
        Initialize the data transformation method.

        :param y_true: Ground-truth labels.
        :param y_pred: Hard labels for model predictions.
        :param y_score: Soft labels for model predictions.
        """
        raise NotImplementedError
