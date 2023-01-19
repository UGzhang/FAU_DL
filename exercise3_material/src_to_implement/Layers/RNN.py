import numpy as np
import copy
from Layers.Base import BaseLayer
from Layers.Sigmoid import Sigmoid
from Layers.TanH import TanH
from Layers.FullyConnected import FullyConnected


class RNN(BaseLayer):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.trainable = True
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.memorize = False
        self.hidden_state = np.zeros(self.hidden_size)

        self.tanh = TanH()
        self.sigmoid = Sigmoid()
        self.optimizer = None
        self.grad_weight1 = None
        self.grad_weight2 = None
        self.FC1 = FullyConnected(self.hidden_size + self.input_size, self.hidden_size)
        self.FC2 = FullyConnected(self.hidden_size, self.output_size)

        self.FC1_input = []
        self.FC2_input = []
        self.Sig_output = []
        self.TanH_output = []

        self.batch_size = None
        self.y_t = None

    def forward(self, input_tensor):
        self.batch_size = len(input_tensor)
        self.y_t = np.zeros((self.batch_size, self.output_size))
        if not self.memorize:
            self.hidden_state = np.zeros(self.hidden_size)

        for i in range(len(input_tensor)):
            xt_hat = np.concatenate((self.hidden_state.reshape(self.hidden_size, 1), input_tensor[i].reshape(input_tensor.shape[1], 1)))
            FC1 = self.FC1.forward(xt_hat.T) # xt_hat * W_h
            self.FC1_input.append(self.FC1.input_tensor)

            self.hidden_state = self.tanh.forward(FC1)  # ht
            self.TanH_output.append(self.tanh.activation)

            FC2 = self.FC2.forward(self.hidden_state)  # ht * W_hy
            self.FC2_input.append(self.FC2.input_tensor)

            self.y_t[i, :] = self.sigmoid.forward(FC2)
            self.Sig_output.append(self.sigmoid.activation)

        return self.y_t

    def backward(self, error_tensor):
        prev_err = np.zeros((self.batch_size, self.input_size))
        hidden_err = np.zeros(self.hidden_size)
        self.grad_weight1 = 0
        self.grad_weight2 = 0

        for i in range(self.batch_size-1, -1, -1):
            self.sigmoid.activation = self.Sig_output[i]
            sig_err = self.sigmoid.backward(error_tensor[i])

            self.FC2.input_tensor = self.FC2_input[i]
            FC2 = self.FC2.backward(sig_err)
            self.grad_weight2 += self.FC2.gradient_weights

            self.tanh.activation = self.TanH_output[i]
            FC1 = self.tanh.backward(FC2 + hidden_err)

            self.FC1.input_tensor = self.FC1_input[i]
            fc1_err = self.FC1.backward(FC1)
            self.grad_weight1 += self.FC1.gradient_weights

            hidden_err = np.squeeze(fc1_err.T[0:self.hidden_size])
            prev_err[i,:] = np.squeeze(np.split(fc1_err.T, [self.hidden_size])[1])

        self.weights = self.FC1.weights

        if self.optimizer is not None:
            self.FC1.weights = self.optimizer.calculate_update(self.FC1.weights, self.grad_weight1)
            self.FC2.weights = self.optimizer.calculate_update(self.FC2.weights, self.grad_weight2)

        return prev_err

    @property
    def gradient_weights(self):
        return self.grad_weight1

    @gradient_weights.setter
    def gradient_weights(self, grad_weights):
        self.grad_weight1 = grad_weights

    @property
    def weights(self):
        return self.FC1.weights

    @weights.setter
    def weights(self, weights):
        self.FC1.weights = weights

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, opt):
        self._optimizer = copy.deepcopy(opt)

    def initialize(self, weights_initializer, bias_initializer):
        if weights_initializer is not None and bias_initializer is not None:
            self.FC1.initialize(weights_initializer, bias_initializer)
            self.FC2.initialize(weights_initializer, bias_initializer)

    @property
    def memorize(self):
        return self._memorize

    @memorize.setter
    def memorize(self, x):
        self._memorize = x


