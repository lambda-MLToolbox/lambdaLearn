from sklearn.pipeline import Pipeline

from lambdaLearn.Transform.Graph.NormalizeFeatures import NormalizeFeatures


class GraphMixin:
    def __init__(self):
        pass

    def init_default_transforms(self):
        """
        Initialize the data transformation method.
        """
        self.transforms = None
        self.target_transform = None
        self.transform = Pipeline([("NormalizeFeatures", NormalizeFeatures())])
        self.unlabeled_transform = Pipeline([("NormalizeFeatures", NormalizeFeatures())])
        self.test_transform = Pipeline([("NormalizeFeatures", NormalizeFeatures())])
        self.valid_transform = Pipeline([("NormalizeFeatures", NormalizeFeatures())])
        return self
