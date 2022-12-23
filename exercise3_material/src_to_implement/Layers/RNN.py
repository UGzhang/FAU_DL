import numpy as np
from Layers.Base import BaseLayer, Tool


class RNN(BaseLayer):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.trainable = True

        self.hidden_state = np.zeros(hidden_size)
        self.memorize = False

    def forward(self, input_tensor):
        pass

    @property
    def memorize(self):
        return self.memorize

    @memorize.setter
    def memorize(self, x):
        self.memorize = x
