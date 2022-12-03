import numpy as np
import math


class Sgd:
    def __init__(self, lr:float):
        self.lr = lr

    def calculate_update(self, weight_tensor, gradient_tensor):
        return weight_tensor - self.lr * gradient_tensor


class SgdWithMomentum:

    def __init__(self, lr, mr):
        self.__learning_rate = lr
        self.__momentum_rate = mr
        self.__v = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        self.__v = self.__momentum_rate * self.__v - self.__learning_rate * gradient_tensor
        return weight_tensor + self.__v


class Adam:

    def __init__(self, lr, mu, rho):
        self.__learning_rate = lr
        self.__mu = mu
        self.__rho = rho

        self.__v = 0
        self.__r = 0
        self.__k = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        self.__k += 1
        self.__v = self.__mu * self.__v + (1 - self.__mu) * gradient_tensor
        self.__r = self.__rho * self.__r + (1-self.__rho) * gradient_tensor * gradient_tensor
        v_hat = self.__v / (1 - math.pow(self.__mu, self.__k))
        r_hat = self.__r / (1 - math.pow(self.__rho, self.__k))
        w = weight_tensor - self.__learning_rate * v_hat / (math.sqrt(r_hat) + np.finfo(float).eps)
        return w

