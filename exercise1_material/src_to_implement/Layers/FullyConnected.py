import numpy as np
from Layers.Base import BaseLayer


class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.trainable = True

        self.weight = np.random.uniform(size=(input_size, output_size))
        self.bias = np.ones(output_size)
        self.weights = np.vstack((self.weight, self.bias))  # cols concat

        self.gradient_weights = None
        self._optimizer = None
        self._x = None

    '''
    y_hat = X * W
    '''
    def forward(self, input_tensor):
        # x concat cols of 1
        self._x = np.c_[input_tensor, np.ones(input_tensor.shape[0])]
        return np.matmul(self._x, self.weights)

    '''
    En-1 = En * W.T
    '''
    def backward(self, error_tensor):
        self.gradient_weights = np.matmul(self._x.T, error_tensor)
        if self._optimizer is not None:
            self.weights = self._optimizer.calculate_update(self.weights, self.gradient_weights)
        return np.dot(error_tensor, self.weight.T)

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, _opt):
        self._optimizer = _opt
