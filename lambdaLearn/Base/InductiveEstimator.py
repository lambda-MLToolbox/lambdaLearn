from abc import abstractmethod

from lambdaLearn.Base.SemiEstimator import SemiEstimator


class InductiveEstimator(SemiEstimator):
    __semi_type__='Inductive'
    @abstractmethod
    def predict(self,X):
    # >> predict(X): Make predictions on the new data.
    # >> - X: Samples to be predicted.
        raise NotImplementedError(
            "Predict method must be implemented."
        )