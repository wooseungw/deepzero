import numpy as np
class MeanSquaredError:
    def __call__(self, y_true, y_pred):
        return ((y_true - y_pred) ** 2).mean()

class CrossEntropy:
    def __call__(self, y_true, y_pred):
        epsilon = 1e-12
        y_pred = y_pred.clip(epsilon, 1 - epsilon)
        return - (y_true * np.log(y_pred)).mean()

def binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-12
    y_pred = y_pred.clip(epsilon, 1 - epsilon)
    return - (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)).mean()

def categorical_cross_entropy(y_true, y_pred):
    epsilon = 1e-12
    y_pred = y_pred.clip(epsilon, 1 - epsilon)
    return - np.sum(y_true * np.log(y_pred), axis=1).mean()