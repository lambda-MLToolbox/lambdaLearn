from abc import abstractmethod

from lambdaLearn.Base.SemiEstimator import SemiEstimator


class TransductiveEstimator(SemiEstimator):
    __semi_type__ = "Transductive"

    @abstractmethod
    def predict(self, X=None, Transductive=True):
        """
        Output the result of transductive learning or make predictions on the new data.
        :param X: The samples to be predicted. It is only valid when Transductive is False.
        :param Transductive: Whether to use transductive learning mechanism to directly output the prediction result of unlabeled_X input during fit.
        """
        raise NotImplementedError("Predict method must be implemented.")
