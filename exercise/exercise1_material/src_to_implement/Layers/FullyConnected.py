import numpy as np

from Base import BaseLayer


class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        super().trainable = True

        self.weight = np.random.uniform(size=(input_size, output_size))
        self.bias = np.ones(output_size)
        self.weights = np.c_(self.weight, self.bias)

        self.gradient_weights = None
        self._optimizer = None
        self._x = None

    def forward(self, input_tensor):
        X = input_tensor
        one = np.ones(X.shapr[0])
        self._x = np.c_[X, one]
        Y_T = np.matmul(self._x, self.weights)
        return Y_T

    def backward(self, error_tensor):
        gradient_error = np.matmul(error_tensor, self.weights)
        self.gradient_weights = np.matmul(self._x.T, error_tensor)
        if self._optimizer is not None:
            self.weights = self._optimizer.calculate_update(self.weights, self.gradient_weights)
        return gradient_error

    @property
    def optimizer(self):
        return self._optimizer()

    @optimizer.setter
    def optimizer(self, _opt):
        self._optimizer = _opt


