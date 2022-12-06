import numpy as np
import math, copy
from Layers.Base import BaseLayer, Tool


class Conv(BaseLayer):

    def __init__(self, stride_shape, convolution_shape, num_kernels):
        super().__init__()
        self.trainable = True

        self.stri_h = stride_shape[0]
        self.stri_w = stride_shape[1] if type(stride_shape) == tuple else 1

        if len(convolution_shape) == 2: convolution_shape = convolution_shape + (1,)

        self.weight_shape = (num_kernels,) + convolution_shape

        self.convolution_shape = convolution_shape

        self.weights = np.random.uniform(size=(num_kernels, *convolution_shape))  # 0-1
        self.bias = np.random.uniform(size=num_kernels)

        # self.gradient_weights = None
        # self.gradient_bias = None

        self._input_shape = None
        self.num_ker = num_kernels
        self._optimizer = None
        self._Conv1D = False

    '''
        input_tensor: [batch,channel,height,width]
    '''
    def forward(self, input_tensor):

        if len(input_tensor.shape) == 3:
            self._Conv1D = True
            input_tensor = np.expand_dims(input_tensor, axis=3)

        self._input_shape = input_tensor.shape
        N, C, H, W = input_tensor.shape
        num, channels, filter_h, filter_w = self.weights.shape

        # SAME padding
        pad_h1 = int(math.floor((filter_h - 1) / 2))
        pad_h2 = int(math.ceil((filter_h - 1) / 2))
        pad_w1 = int(math.floor((filter_w - 1) / 2))
        pad_w2 = int(math.ceil((filter_w - 1) / 2))
        input_tensor = np.pad(input_tensor, ((0, 0), (0, 0), (pad_h1, pad_h2), (pad_w1, pad_w2)), 'constant', constant_values=0)

        batch, channels, height, width = input_tensor.shape

        self.input_shape = input_tensor.shape
        self.input_channel = channels

        out_h = int((height - filter_h) / self.stri_h) + 1
        out_w = int((width - filter_w) / self.stri_w) + 1

        self.input_col = Tool().im2col(input_tensor, [filter_h, filter_w], [self.stri_h, self.stri_w])
        self.weight_col = self.weights.reshape(self.num_ker, -1).T

        out = np.dot(self.input_col, self.weight_col) + self.bias

        output = out.reshape(batch, out_h, out_w, -1).transpose(0, 3, 1, 2)
        self.output = output

        if self._Conv1D:
            self.output = np.squeeze(self.output, axis=3)
        return self.output

    def backward(self, error_tensor):
        error_tensor = error_tensor.transpose(0, 2, 3, 1).reshape(-1, self.num_ker)

        self.gradient_weights = np.dot(self.input_col.T, error_tensor)
        self.gradient_weights = self.gradient_weights.transpose(1, 0).reshape(self.weights.shape)
        self.gradient_bias = np.sum(error_tensor, axis=0)
        num, channels, filter_h, filter_w = self.weights.shape

        dcol = np.dot(error_tensor, self.weight_col.T)
        dx = Tool().col2im(dcol, self._input_shape, [filter_h,filter_w], [self.stri_h, self.stri_w])

        if self._optimizer is not None:
            self.weights = self._optimizer.calculate_update(self.weights, self.gradient_weights)
            self.bias = self._optimizer.calculate_update(self.bias, self.gradient_bias)
        return dx

    def initialize(self, weights_initializer, bias_initializer):
        self.weights = weights_initializer.initialize(self.weights.shape, np.prod(self.convolution_shape),
                                                      np.prod(self.convolution_shape[1:]) * self.num_kernels)
        self.bias = bias_initializer.initialize(self.bias.shape, 1, self.num_kernels)


    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, _opt):
        self._optimizer = _opt
        self._optimizer.weights = copy.deepcopy(_opt)
        self._optimizer.bias = copy.deepcopy(_opt)
