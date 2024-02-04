import os
import numpy as np
from environment.domain import Ball
from learner.models.dynamic.ader import Ader
from utils.data_generator import LinearRegressionGenerator
from utils.plot import plot
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
'''
We demonstrate how to use the non-stationary online learning methods in LambdaLearn Toolbox.

Specifically, we present two styles to utilize our methods:
 - The first approach aligns with the classic online learning protocol, wherein the model updates in online fashion.
 - The second approach conforms to the offline learning paradigm, where we offer online-to-batch conversion, enabling the model to ingest offline data as input.
'''

# Case i: online learning fashion

# Prepare model
T, dimension, stage, R, Gamma, scale = 2000, 3, 100, 1, 1, 1/2
feature, label = LinearRegressionGenerator().generate_data(
    T, dimension, stage, R, Gamma, seed = 0)
D, r = 2 * R, R
G = scale * D * Gamma ** 2
C = scale * 1 / 2 * (D * Gamma)**2
L_smooth = Gamma**2

domain = Ball(dimension=dimension, radius=R)
min_step_size, max_step_size = D / (G * T**0.5), D / G
online_learner =  Ader(
                domain=domain,
                T=T,
                G=G,
                surrogate=False,
                min_step_size=min_step_size,
                max_step_size=max_step_size,
                seed=0) 
labels = 'Ader'

# Online interaction
loss = np.zeros((T))
for t in range(T):
    # Decision
    x_t = online_learner.predict()
    # Get loss function
    loss_func = lambda x: scale * 1 / 2 * ((np.dot(x, feature[t]) - label[t])**2)
    loss[t] = loss_func(x_t)
    # Update model
    online_learner.fit(loss_func = loss_func, online_to_batch = False)
    
# Plot loss curve
if os.path.exists('./results') is False:
    os.makedirs('./results')
plot(loss, labels, file_path='./results/full_dynamic_case1.pdf')


# Case ii: online to batch conversion, where the data is given in the offline fashion.

# Preprocess dataset 
X, y = fetch_california_housing(return_X_y = True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, shuffle = False)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Prepare model
T, dimension, R, Gamma, scale = X_train.shape[0], X.shape[1], 1, 1, 1 / 2
D, r = 2 * R, R
G = scale * D * Gamma ** 2
C = scale * 1 / 2 * (D * Gamma)**2
L_smooth = Gamma**2
domain = Ball(dimension=dimension, radius=R)
min_step_size, max_step_size = D / (G * T**0.5), D / G 
online_learner = Ader(domain = domain, 
                      T = T, 
                      G = G, 
                      surrogate = False,
                      min_step_size = min_step_size, 
                      max_step_size = max_step_size, 
                      seed = 0)
labels = 'Ader'

# Train 
train_loss = np.zeros(y_train.shape)
loss_func = lambda x, feature, label: scale * 1 / 2 * ((np.dot(x, feature) - label)**2)
online_learner, train_loss = online_learner.fit(X_train, y_train, online_to_batch = True, loss_func = loss_func)

# Test 
test_loss = np.zeros(y_test.shape)
for idx in range(X_test.shape[0]):
    theta = online_learner.predict(X_test[idx])
    test_loss[idx] = loss_func(theta, X_test[idx], y_test[idx])
print(f'Tese MSE: {test_loss.mean()}')

# Plot training curve
plot(train_loss, labels, file_path='./results/full_dynamic_case2.pdf')
   
