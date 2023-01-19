import numpy as np
import copy
from Layers.Base import BaseLayer
from Layers.Helpers import compute_bn_gradients


class BatchNormalization(BaseLayer):
    def __init__(self, channels):
        super().__init__()
        self.trainable = True
        self._channels = channels

        self.bias = np.zeros(self._channels)
        self.weights = np.ones(self._channels)
        self.gradient_bias = None
        self.gradient_weights = None

        self._mean = None
        self._variance = None

        self._decay = 0.8
        self._moving_mean = None
        self._moving_variance = None

        self._x_hat = None
        self._optimizer = None

        self._original_input_shape = None
        self._input = None

    def initialize(self, weights_initializer, bias_initializer):
        self.bias = bias_initializer.initialize(self._channels)
        self.weights = weights_initializer.initialize(self._channels, self._channels, self._channels)

    def forward(self, input_tensor):
        self._original_input_shape = input_tensor.shape
        if len(self._original_input_shape) == 4:
            input_tensor = self.reformat(input_tensor)
        self._input = input_tensor
        # test
        if self.testing_phase:
            self._mean = self._moving_mean
            self._variance = self._moving_variance
        # train
        else:
            self._mean = np.mean(input_tensor, axis=0)
            self._variance = np.var(input_tensor, axis=0)
            if self._moving_mean is None or self._moving_variance is None:
                self._moving_mean = copy.deepcopy(self._mean)
                self._moving_variance = copy.deepcopy(self._variance)
            else:
                self._moving_mean = self._decay * self._moving_mean + (1 - self._decay) * self._mean
                self._moving_variance = self._decay * self._moving_variance + (1 - self._decay) * self._variance
        self._x_hat = (input_tensor - self._mean) / (np.sqrt(self._variance + np.finfo(float).eps))
        Y = self.weights * self._x_hat + self.bias
        if len(self._original_input_shape) == 4:
            Y = self.reformat(Y)
        return Y

    def backward(self, error_tensor):
        E = error_tensor
        if E.ndim == 4:
            E = self.reformat(E)
        self.gradient_weights = np.sum(E * self._x_hat, axis=0)
        self.gradient_bias = np.sum(E, axis=0)
        grad_input = compute_bn_gradients(E, self._input, self.weights, self._mean, self._variance)
        if self._optimizer is not None:
            self.weights = self._optimizer.weights.calculate_update(self.weights, self.gradient_weights)
            self.bias = self._optimizer.bias.calculate_update(self.bias, self.gradient_bias)
        if len(self._original_input_shape) == 4:
            grad_input = self.reformat(grad_input)
        return grad_input

    def reformat(self, tensor):
        # 4-d to 2-d
        if tensor.ndim == 4:
            B, H, M, N = tensor.shape
            new_tensor = tensor.reshape(B, H, M * N).transpose(0, 2, 1).reshape(B * M * N, H)
            return new_tensor
        # 2-d to 4-d
        elif tensor.ndim == 2:
            B, H, M, N = self._original_input_shape
            new_tensor = tensor.reshape(B, M * N, H).transpose(0, 2, 1).reshape(B, H, M, N)
            return new_tensor

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, _opt):
        self._optimizer = _opt
        self._optimizer.weights = copy.deepcopy(_opt)
        self._optimizer.bias = copy.deepcopy(_opt)