import numpy as np
from Layers.Base import BaseLayer
from Layers.FullyConnected import FullyConnected
from Layers.TanH import TanH


class RNN(BaseLayer):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.trainable = True
        self._memorize = False

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.FC_h = FullyConnected(input_size+hidden_size, hidden_size)
        self.FC_y = FullyConnected(hidden_size, output_size)

        self.h_t = None
        self.prev_h_t = None

    def forward(self, input_tensor):
        batch_size = input_tensor.shape[0]
        if self._memorize and self.h_t:
            self.h_t[0] = self.prev_h_t
        else:
            self.h_t = np.zeros((batch_size+1, self.hidden_size))
        y_t = np.zeros((batch_size, self.output_size))
