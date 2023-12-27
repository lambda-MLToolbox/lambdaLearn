import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline

from lambdaLearn.Transform.ToImage import ToImage
from lambdaLearn.Transform.ToTensor import ToTensor
from lambdaLearn.Transform.Vision.Normalization import Normalization


class VisionMixin:
    def __init__(self,mean=None,std=None):
        self.mean=mean
        self.std=std
        # >> Parameter
        # >> - mean: Mean of the dataset.
        # >> - std: Standard deviation of the dataset.
    def init_default_transforms(self):
        # >> init_default_transform: Initialize the default data transformation method.
        self.transforms=None
        self.target_transform=None
        self.pre_transform=ToImage()
        self.transform=Pipeline([('ToTensor',ToTensor(dtype='float',image=True)),
                              ('Normalization',Normalization(mean=self.mean,std=self.std))
                              ])
        self.unlabeled_transform=Pipeline([('ToTensor',ToTensor(dtype='float',image=True)),
                              ('Normalization',Normalization(mean=self.mean,std=self.std))
                              ])
        self.test_transform=Pipeline([('ToTensor',ToTensor(dtype='float',image=True)),
                              ('Normalization',Normalization(mean=self.mean,std=self.std))
                              ])
        self.valid_transform=Pipeline([('ToTensor',ToTensor(dtype='float',image=True)),
                              ('Normalization',Normalization(mean=self.mean,std=self.std))
                              ])
        return self


    def show_image(self,img):
        plt.imshow(img)
        plt.axis('off')
        plt.show()
        return img