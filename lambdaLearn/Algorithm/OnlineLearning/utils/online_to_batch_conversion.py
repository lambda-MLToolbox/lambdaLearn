from typing import Union

import numpy as np

from environment.loss_function import SquareLoss, CustomFunction
from environment.environment import Environment


def online_to_batch_conversion(learner, X, y, loss_func = None):
    # loss/criterion (default: square loss)
    if loss_func == None:
        loss_func = SquareLoss(feature = X, label = y, scale = 0.5)
    else:
        loss_func = CustomFunction(feature = X, label = y, f = loss_func)
    env = Environment(func_sequence=loss_func)
    T = X.shape[0]
    if hasattr(learner, 'domain'):
        dimension = learner.domain.dimension
    else:
        dimension = learner.schedule.bases[0].domain.dimension
    x = np.zeros((T, dimension))
    loss, surrogate_loss = np.zeros(T), np.zeros(T)
    # 训练T轮
    for t in range(T):
        x_submit = learner.predict(X[t])
        x[t], loss[t], surrogate_loss[t] = learner.opt(env[t])
        # callback
        if t%100 == 0:
            print(f'step: {t} - loss: {loss[t]}')


    return x, loss, surrogate_loss