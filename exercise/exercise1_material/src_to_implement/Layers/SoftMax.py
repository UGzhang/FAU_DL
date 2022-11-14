from Layers.Base import BaseLayer
import numpy as np


class SoftMax(BaseLayer):
    def __init__(self):
        super().__init__()
        self._y = None

    def forward(self, input_tensor):
        input_tensor -= np.max(input_tensor, axis=1, keepdims=True)
        self._y = np.exp(input_tensor) / np.sum(np.exp(input_tensor), axis=1, keepdims=True)
        return self._y

    def backward(self, error_tensor):
        num_row = error_tensor.shape[0]
        num_column = error_tensor.shape[1]
        # calculate E_n-1 step-by-step according to slide 16
        weighted_sum = np.sum(self._y * error_tensor, axis=1).repeat(num_column).reshape((num_row, num_column))
        result_e = self._y * (error_tensor - weighted_sum)
        return result_e

