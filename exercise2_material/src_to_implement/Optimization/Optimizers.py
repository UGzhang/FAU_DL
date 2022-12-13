import numpy as np
import math


class Sgd:
    def __init__(self, lr: float):
        self.lr = lr

    def calculate_update(self, weight_tensor, gradient_tensor):
        return weight_tensor - self.lr * gradient_tensor


class SgdWithMomentum:

    def __init__(self, lr, mr):
        self._lr = lr
        self._mr = mr  # momentum rate
        self._v = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        self._v = self._mr * self._v - self._lr * gradient_tensor
        return weight_tensor + self._v


class Adam:

    def __init__(self, lr, mu, rho):
        self._lr = lr
        self._mu = mu
        self._rho = rho
        self._v = 0
        self._r = 0
        self._k = 1

    def calculate_update(self, weight_tensor, gradient_tensor):
        self._v = self._mu * self._v + (1 - self._mu) * gradient_tensor
        self._r = self._rho * self._r + (1 - self._rho) * gradient_tensor * gradient_tensor
        v_hat = self._v / (1 - self._mu ** self._k)
        r_hat = self._r / (1 - self._rho ** self._k)
        w = weight_tensor - self._lr * (v_hat / (np.sqrt(r_hat) + np.finfo(float).eps))
        self._k += 1
        return w
