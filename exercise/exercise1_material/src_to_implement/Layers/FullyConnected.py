import numpy as np
from Layers.Base import BaseLayer


class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.trainable = True

        self.weight = np.random.uniform(size=(input_size, output_size))
        self.bias = np.ones(output_size)
        self.weights = np.vstack((self.weight, self.bias))

        self.gradient_weights = None
        self._optimizer = None
        self._x = None

    def forward(self, input_tensor):
        X = input_tensor
        one = np.ones(X.shape[0])
        self._x = np.c_[X, one]
        Y_T = np.matmul(self._x, self.weights)
        return Y_T

    def backward(self, error_tensor):
        gradient_error = np.dot(error_tensor, self.weight.T)
        self.gradient_weights = np.matmul(self._x.T, error_tensor)
        if self._optimizer is not None:
            self.weights = self._optimizer.calculate_update(self.weights, self.gradient_weights)
        return gradient_error

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, _opt):
        self._optimizer = _opt
