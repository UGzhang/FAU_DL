import math

import numpy as np
import copy
from Layers.Base import BaseLayer
from Layers.Helpers import compute_bn_gradients


class BatchNormalization(BaseLayer):
    def __init__(self, channels):
        super().__init__()
        self.trainable = True
        self.channels = channels
        self.bias = None
        self.weights = None

        self.mean = None
        self.variance = None

        self.decay = 0.8
        self.moving_mean = None
        self.moving_variance = None

        self.x_hat = None
        self._optimizer = None

        self.input_shape = None

    def initialize(self):
        self.weights = np.ones(self.channels)
        self.bias = np.zeros(self.channels)

    def forward(self, input_tensor):
        self.input_shape = input_tensor.shape
        X = input_tensor
        if self.input_shape.ndim == 4:
            X = self.reformat(X)
        self.X = X
        # test
        if self.testing_phase:
            self.mean = self.moving_mean
            self.variance = self.moving_variance
        # train
        else:
            self.mean = np.mean(X, axis=0)
            self.variance = np.var(X, axis=0)
            if self.moving_mean is None or self.moving_variance is None:
                self.moving_mean = copy.deepcopy(self.mean)
                self.moving_variance = copy.deepcopy(self.variance)
            else:
                self.moving_mean = self.decay * self.moving_mean + (1 - self.decay) * self.mean
                self.moving_variance = self.decay * self.moving_variance + (1 - self.decay) * self.variance
        self.x_hat = (X - self.mean) / (math.sqrt(self.variance + np.finfo(float).eps))
        Y = self.weights * self.x_hat + self.bias
        if self.input_shape.ndim == 4:
            Y = self.reformat(Y)
        return Y

    def backward(self, error_tensor):
        E = error_tensor
        if E.ndim == 4:
            E = self.reformat(E)
        grad_weights = np.sum(E * self.x_hat, axis=0)
        grad_bias = np.sum(E, axis=0)
        grad_input = compute_bn_gradients(E, self.X, self.weights, self.mean, self.variance)
        if self._optimizer is not None:
            self._optimizer.weights.calculate_update(self.weights, grad_weights)
            self._optimizer.bias.calculate_update(self.bias, grad_bias)
        if self.input_shape.ndim == 4:
            grad_input = self.reformat(grad_input)
        return grad_input

    def reformat(self, tensor):
        # 4-d to 2-d
        if tensor.ndim == 4:
            B, H, M, N = tensor.shape
            new_tensor = tensor.reshape(B, H, M * N).transpose(0, 2, 1).reshape(B * M * N, H)
            return new_tensor
        # 2-d to 4-d
        elif tensor.ndim == 2:
            B, H, M, N = self.input_shape
            new_tensor = tensor.reshape(B, M * N, H).transpose(0, 2, 1).reshape(B, H, M, N)
            return new_tensor

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, _opt):
        self._optimizer = _opt
        self._optimizer.weights = copy.deepcopy(_opt)
        self._optimizer.bias = copy.deepcopy(_opt)