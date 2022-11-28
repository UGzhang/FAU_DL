import numpy as np
from Layers.Base import BaseLayer


class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.trainable = True

        self.weight = np.random.uniform(size=(input_size, output_size))
        self.bias = np.ones(output_size)
        self.weights = np.vstack((self.weight, self.bias))  # cols concat

        self.gradient_weights = None
        self.__optimizer = None
        self.__x = None

        # add in ex2
        self.__input_size = input_size
        self.__output_size = output_size

    '''
    y_hat = X * W
    '''
    def forward(self, input_tensor):
        # x concat cols of 1
        self.__x = np.c_[input_tensor, np.ones(input_tensor.shape[0])]
        return np.matmul(self.__x, self.weights)

    '''
    En-1 = En * W.T
    '''
    def backward(self, error_tensor):
        self.gradient_weights = np.matmul(self.__x.T, error_tensor)
        if self.__optimizer is not None:
            self.weights = self.__optimizer.calculate_update(self.weights, self.gradient_weights)
        return np.dot(error_tensor, self.weight.T)

    @property
    def optimizer(self):
        return self.__optimizer

    @optimizer.setter
    def optimizer(self, _opt):
        self.__optimizer = _opt

    # add in ex2
    def initialize(self, weights_initializer, bias_initializer):
        self.weight = weights_initializer.initialize(self.weights.shape, self.__input_size, self.__output_size)
        self.bias = bias_initializer.initialize(self.weights.shape, self.__input_size, self.__output_size)
