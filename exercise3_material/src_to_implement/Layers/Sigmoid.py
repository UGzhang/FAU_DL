from Layers.Base import BaseLayer
import numpy as np


class Sigmoid(BaseLayer):
    def __init__(self):
        super().__init__()
        self._x = None  # f(x)

    def forward(self, input_tensor):
        self._x = 1 / (1 + np.exp(-input_tensor))
        return self._x

    def backward(self, error_tensor):
        return self._x * (1 - self._x) * error_tensor
