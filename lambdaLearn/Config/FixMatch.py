from lambdaLearn.Augmentation.Vision.RandomHorizontalFlip import RandomHorizontalFlip
from lambdaLearn.Augmentation.Vision.RandomCrop import RandomCrop
from lambdaLearn.Augmentation.Vision.RandAugment import RandAugment
from lambdaLearn.Augmentation.Vision.Cutout import Cutout
from lambdaLearn.Opitimizer.SGD import SGD
from lambdaLearn.Scheduler.CosineAnnealingLR import CosineAnnealingLR
from lambdaLearn.Network.WideResNet import WideResNet
from lambdaLearn.Dataloader.UnlabeledDataloader import UnlabeledDataLoader
from lambdaLearn.Dataloader.LabeledDataloader import LabeledDataLoader
from lambdaLearn.Sampler.RandomSampler import RandomSampler
from lambdaLearn.Sampler.SequentialSampler import SequentialSampler
from sklearn.pipeline import Pipeline
from lambdaLearn.Evaluation.Classifier.Accuracy import Accuracy
from lambdaLearn.Evaluation.Classifier.Top_k_Accuracy import Top_k_Accurary
from lambdaLearn.Evaluation.Classifier.Precision import Precision
from lambdaLearn.Evaluation.Classifier.Recall import Recall
from lambdaLearn.Evaluation.Classifier.F1 import F1
from lambdaLearn.Evaluation.Classifier.AUC import AUC
from lambdaLearn.Evaluation.Classifier.Confusion_Matrix import Confusion_Matrix
from lambdaLearn.Dataset.LabeledDataset import LabeledDataset
from lambdaLearn.Dataset.UnlabeledDataset import UnlabeledDataset
from lambdaLearn.Transform.Vision.Normalization import Normalization
from lambdaLearn.Transform.ToTensor import ToTensor
from lambdaLearn.Transform.ToImage import ToImage

mean = [0.4914, 0.4822, 0.4465]
std = [0.2471, 0.2435, 0.2616]

pre_transform = ToImage()
transforms = None
target_transform = None
transform = Pipeline([('ToTensor', ToTensor(dtype='float',image=True)),
                    ('Normalization', Normalization(mean=mean, std=std))
                    ])
unlabeled_transform = Pipeline([('ToTensor', ToTensor(dtype='float',image=True)),
                                ('Normalization', Normalization(mean=mean, std=std))
                                ])
test_transform = Pipeline([('ToTensor', ToTensor(dtype='float',image=True)),
                                ('Normalization', Normalization(mean=mean, std=std))
                                ])
valid_transform = Pipeline([('ToTensor', ToTensor(dtype='float',image=True)),
                                 ('Normalization', Normalization(mean=mean, std=std))
                                 ])

train_dataset=None
labeled_dataset=LabeledDataset(pre_transform=pre_transform,transforms=transforms,
                               transform=transform,target_transform=target_transform)

unlabeled_dataset=UnlabeledDataset(pre_transform=pre_transform,transform=unlabeled_transform)

valid_dataset=UnlabeledDataset(pre_transform=pre_transform,transform=valid_transform)

test_dataset=UnlabeledDataset(pre_transform=pre_transform,transform=test_transform)

# Batch sampler
train_batch_sampler=None
labeled_batch_sampler=None
unlabeled_batch_sampler=None
valid_batch_sampler=None
test_batch_sampler=None

# sampler
train_sampler=None
labeled_sampler=RandomSampler(replacement=True,num_samples=64*(2**20))
unlabeled_sampler=RandomSampler(replacement=True)
valid_sampler=SequentialSampler()
test_sampler=SequentialSampler()

#dataloader
train_dataloader=None
labeled_dataloader=LabeledDataLoader(batch_size=64,num_workers=0,drop_last=True)
unlabeled_dataloader=UnlabeledDataLoader(num_workers=0,drop_last=True)
valid_dataloader=UnlabeledDataLoader(batch_size=64,num_workers=0,drop_last=False)
test_dataloader=UnlabeledDataLoader(batch_size=64,num_workers=0,drop_last=False)

# network
network=WideResNet(num_classes=10,depth=28,widen_factor=2,drop_rate=0)

# optimizer
optimizer=SGD(lr=0.03,momentum=0.9,nesterov=True)

# scheduler
scheduler=CosineAnnealingLR(eta_min=0,T_max=2**20)

# augmentation
weak_augmentation=Pipeline([('RandomHorizontalFlip',RandomHorizontalFlip()),
                              ('RandomCrop',RandomCrop(padding=0.125,padding_mode='reflect')),
                              ])

strong_augmentation=Pipeline([('RandomHorizontalFlip',RandomHorizontalFlip()),
                              ('RandomCrop',RandomCrop(padding=0.125,padding_mode='reflect')),
                              ('RandAugment',RandAugment(n=2,m=10,num_bins=10)),
                              ('Cutout',Cutout(v=0.5,fill=(127, 127, 127))),
                              ])
augmentation={
    'weak_augmentation':weak_augmentation,
    'strong_augmentation':strong_augmentation
}

# evalutation
evaluation={
    'accuracy':Accuracy(),
    'top_5_accuracy':Top_k_Accurary(k=5),
    'precision':Precision(average='macro'),
    'Recall':Recall(average='macro'),
    'F1':F1(average='macro'),
    'AUC':AUC(multi_class='ovo'),
    'Confusion_matrix':Confusion_Matrix(normalize='true')
}

# model
weight_decay=5e-4
ema_decay=0.999
epoch=1
num_it_total=2**20
num_it_epoch=2**20
eval_epoch=None
eval_it=None
device='cpu'

parallel=None
file=None
verbose=False

threshold=0.95
lambda_u=1
T=0.5
mu=7