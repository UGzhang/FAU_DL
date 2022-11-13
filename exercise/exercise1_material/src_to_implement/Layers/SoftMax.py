from Base import BaseLayer
import numpy as np


class SoftMax(BaseLayer):
    def __init__(self):
        super.__init__()
        self._y = None

    def forward(self, input_tensor):
        input_tensor -= np.max(input_tensor)
        self._y = np.exp(input_tensor) / np.sum(np.exp(input_tensor), axis=0)
        return self._y

    def backward(self, error_tensor):
        return self._y * (error_tensor - np.sum(error_tensor * self._y))


