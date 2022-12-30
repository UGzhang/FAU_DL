import numpy as np
import copy
from Layers.Base import BaseLayer


class RNN(BaseLayer):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.trainable = True

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.hidden_state = np.zeros(hidden_size)
        self.memorize = False

        self.W_y = np.random.uniform(size=(hidden_size, output_size))
        self.B_y = np.random.uniform(size=output_size)
        self.W_hx = np.random.uniform(size=(input_size + hidden_size, hidden_size))
        self.B_hx = np.random.uniform(size=hidden_size)

        self._optimizer = None

    def forward(self, input_tensor):
        batch_size = input_tensor.shape[0]
        output_tensor = np.zeros((batch_size, self.hidden_size))
        if self.memorize is False:
            self.hidden_state = np.zeros(self.hidden_size)
        for b in range(batch_size):
            x_hat = np.hstack((self.hidden_state, input_tensor[b]))
            self.hidden_state = np.tanh(np.dpt(x_hat, self.W_hx) + self.B_hx)  # ht
            y = np.dot(self.hidden_state, self.W_y) + self.B_y
            output_tensor[b] = 1 / (1 + np.exp(-y))

        return output_tensor

    def initialize(self, weights_initializer, bias_initializer):
        self.W_y = weights_initializer.initialize(self.W_y.shape, self.hidden_size, self.output_size)
        self.B_y = bias_initializer.initialize(self.B_y.shape, self.hidden_size, self.output_size)
        self.W_hx = weights_initializer.initialize(self.W_hx.shape, self.W_hx.shape[0], self.hidden_size)
        self.B_hx = bias_initializer.initialize(self.B_hx.shape, self.W_hx.shape[0], self.hidden_size)

    @property
    def calculate_regularization_loss(self):
        return self.optimizer.regularizer.norm(self.)

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, _opt):
        self._optimizer = _opt
        # self._optimizer.weights = copy.deepcopy(_opt)
        # self._optimizer.bias = copy.deepcopy(_opt)

    @property
    def memorize(self):
        return self.memorize

    @memorize.setter
    def memorize(self, x):
        self.memorize = x