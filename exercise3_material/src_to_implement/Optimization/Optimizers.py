import numpy as np
from Optimization.Constraints import L1_Regularizer, L2_Regularizer

# add by ex3
class Optimizer:
    def __init__(self):
        self.regularizer = None

    def add_regularizer(self, regularizer):
        self.regularizer = regularizer


class Sgd(Optimizer):
    def __init__(self, lr: float):
        super().__init__()
        self._lr = lr

    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.regularizer is not None:
            return weight_tensor - self._lr * self.regularizer.calculate_gradient(weight_tensor) - self._lr * gradient_tensor
        else:
            return weight_tensor - self._lr * gradient_tensor


class SgdWithMomentum(Optimizer):
    def __init__(self, lr, mr):
        super().__init__()
        self._lr = lr
        self._mr = mr  # momentum rate
        self._v = 0  # the previous gradient

    def calculate_update(self, weight_tensor, gradient_tensor):
        self._v = self._mr * self._v - self._lr * gradient_tensor
        if self.regularizer is not None:
            return weight_tensor + self._v - self._lr * self.regularizer.calculate_gradient(weight_tensor)
        else:
            return weight_tensor + self._v


class Adam(Optimizer):
    def __init__(self, lr, mu, rho):
        super().__init__()
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
        if self.regularizer is not None:
            return w - self._lr * self.regularizer.calculate_gradient(weight_tensor)
        else:
            return w
