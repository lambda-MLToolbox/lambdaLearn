from lambdaLearn.Evaluation.Classifier.Accuracy import Accuracy
from lambdaLearn.Evaluation.Classifier.AUC import AUC
from lambdaLearn.Evaluation.Classifier.Confusion_Matrix import Confusion_Matrix
from lambdaLearn.Evaluation.Classifier.F1 import F1
from lambdaLearn.Evaluation.Classifier.Precision import Precision
from lambdaLearn.Evaluation.Classifier.Recall import Recall

kernel = "rbf"
gamma = 1
n_neighbors = 7
max_iter = 10000
tol = 1e-3
n_jobs = None
evaluation={
    'accuracy':Accuracy(),
    'precision':Precision(average='macro'),
    'Recall':Recall(average='macro'),
    'F1':F1(average='macro'),
    'AUC':AUC(multi_class='ovo'),
    'Confusion_matrix':Confusion_Matrix(normalize='true')
}
verbose = False
file = None