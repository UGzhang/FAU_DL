import numpy as np
from Layers.Base import BaseLayer


class Dropout(BaseLayer):
    def __init__(self, prob):
        super().__init__()
        self.prob = prob
        self.mask = None

    def forward(self, input_tensor):
        # test
        if self.testing_phase:
            return input_tensor
        # train
        else:
            self.mask = (np.random.rand(*input_tensor.shape) < self.prob) / self.prob
            return input_tensor * self.mask

    def backward(self, error_tensor):
        return error_tensor * self.mask


