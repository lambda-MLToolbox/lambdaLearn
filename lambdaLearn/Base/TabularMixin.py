from sklearn import preprocessing
from sklearn.pipeline import Pipeline

from lambdaLearn.Transform.ToTensor import ToTensor


class TabularMixin:
    def __init__(self):
        pass

    def init_default_transforms(self):
        """
        Initialize the default data transformation method.
        """
        self.transforms = None
        self.target_transform = None
        self.pre_transform = Pipeline([("StandardScaler", preprocessing.StandardScaler())])
        self.transform = Pipeline([("ToTensor", ToTensor())])
        self.unlabeled_transform = Pipeline([("ToTensor", ToTensor())])
        self.test_transform = Pipeline([("ToTensor", ToTensor())])
        self.valid_transform = Pipeline([("ToTensor", ToTensor())])
        return self
