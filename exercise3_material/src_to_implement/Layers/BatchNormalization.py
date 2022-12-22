import numpy as np
import copy
from Layers.Base import BaseLayer


class BatchNormalization(BaseLayer):
    def __init__(self, channels):
        super().__init__()
        self.trainable = True
        self.channels = channels
        self.bias = None
        self.weights = None

        self.mean = None
        self.variance = None

        self.decay = 0.8
        self.moving_mean = None
        self.moving_variance = None

        self.input_shape = None

    def initialize(self):
        self.weights = np.ones(self.channels)
        self.bias = np.zeros(self.channels)

    def forward(self, input_tensor):
        X = input_tensor
        if X.ndim == 4:
            X = self.reformat(X)
        # test
        if self.testing_phase:
            pass
        # train
        else:
            self.mean = np.mean(X, axis=0)
            self.variance = np.var(X, axis=0)
            if self.moving_mean is None or self.moving_variance is None:
                self.moving_mean = copy.deepcopy(self.mean)
                self.moving_variance = copy.deepcopy(self.variance)
            else:
                self.moving_mean = self.decay * self.moving_mean + (1 - self.decay) * self.mean
                self.moving_variance = self.decay * self.moving_variance + (1 - self.decay) * self.variance

    def backward(self, error_tensor):
        pass

    def compute_bn_gradients(self, error_tensor, input_tensor, weights, mean, var):
        pass

    def reformat(self, tensor):
        self.input_shape = tensor.shape
        # 4-d to 2-d
        if tensor.ndim == 4:
            B, H, M, N = tensor.shape
            new_tensor = tensor.reshape(B, H, M * N).transpose(0, 2, 1).reshape(B * M * N, H)
            return new_tensor
        # 2-d to 4-d
        elif tensor.ndim == 2:
            B, H, M, N = self.input_shape
            new_tensor = tensor.reshape(B, M * N, H).transpose(0, 2, 1).reshape(B, H, M, N)
            return new_tensor