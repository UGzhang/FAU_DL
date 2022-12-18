import numpy as np


class L1_Regularizer:
    def __init__(self, alpha):
        self._alpha = alpha

    def calculate_gradient(self, weights):
        return self._alpha * np.sign(weights)

    def norm(self, weights):
        return self._alpha * np.sum(np.abs(weights))


class L2_Regularizer:
    def __init__(self, alpha):
        self._alpha = alpha

    def calculate_gradient(self, weights):
        return self._alpha * weights

    def norm(self, weights):
        return self._alpha * np.sum(np.power(np.abs(weights),2))

