import numpy as np
from sklearn.datasets import load_svmlight_file
from poss_mse import *
from poss_rss import *

# load data
X,y = load_svmlight_file('sonar_scale')
X = X.toarray()

# normalization: make all the variables have expectation 0 and variance 1
A = X - np.mean(X, axis=0)
B = A / np.std(A, axis=0)
X = B[:,np.where(np.isnan(B[0,:])==0)[0]]
A = y - np.mean(y, axis=0)
y = A / np.std(A, axis=0)

# set the size constraint k
k = 8

# use the POSS_MSE function to select the variables
selectedVariables_MSE = POSS_MSE(X,y,k)
print(f'selected variables using MSE: {selectedVariables_MSE}')

# set the tradeoff parameter lambda between mean squared error and l_2 norm regularization
lambda_ = 0.9615

# use the POSS_RSS function to select the variables
selectedVariables_RSS = POSS_RSS(X,y,k,lambda_)
print(f'selected variables using RSS: {selectedVariables_RSS}')
