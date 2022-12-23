from Layers.Base import BaseLayer
import numpy as np


class TanH(BaseLayer):
    def __init__(self):
        super().__init__()
        self._activ = None

    def forward(self, input_tensor):
        self._activ = np.tanh(input_tensor)
        return self._activ

    def backward(self, error_tensor):
        return (1 - np.square(self._activ)) * self._activ
