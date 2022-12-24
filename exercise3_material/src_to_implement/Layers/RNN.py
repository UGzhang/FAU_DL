import numpy as np
import copy
from Layers.Base import BaseLayer


class RNN(BaseLayer):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.trainable = True

        self.hidden_state = np.zeros(hidden_size)
        self.memorize = False

        self.input_size = input_size

        self.weights = None
        self.bias = None

        self._optimizer = None

    def forward(self, input_tensor):
        if self.memorize:
            pass
        else:
            pass

    @property
    def memorize(self):
        return self.memorize

    @memorize.setter
    def memorize(self, x):
        self.memorize = x

    def initialize(self, weights_initializer, bias_initializer):
        self.weights = weights_initializer.initialize(self.input_size[0])
        self.bias = bias_initializer.initialize(self.input_size[0])

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, _opt):
        self._optimizer = _opt
        self._optimizer.weights = copy.deepcopy(_opt)
        self._optimizer.bias = copy.deepcopy(_opt)