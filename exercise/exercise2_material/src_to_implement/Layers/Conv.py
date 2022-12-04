import numpy as np
import math
from Layers.Base import BaseLayer, Tool


class Conv(BaseLayer):

    def __init__(self, stride_shape, convolution_shape, num_kernels):
        super().__init__()
        self.trainable = True

        self.stri_h = stride_shape[0]
        self.stri_w = stride_shape[1] if type(stride_shape) == tuple else 1

        if len(convolution_shape) == 2: convolution_shape = convolution_shape + (1,)

        self.weight_shape = (num_kernels,) + convolution_shape

        self.weights = np.random.uniform(size=(num_kernels, *convolution_shape))  # 0-1
        self.bias = np.random.uniform(size=num_kernels)

        self.gradient_weights = None
        self.gradient_bias = None

        self._input_shape = None

    '''
        input_tensor: [batch,channel,height,width]
    '''
    def forward(self, input_tensor):

        self._input_shape = input_tensor.shape
        N, C, H, W = input_tensor.shape
        num, channels, filter_h, filter_w = self.weights.shape

        # SAME padding
        pad_h1 = int(math.floor((filter_h - 1) / 2))
        pad_h2 = int(math.ceil((filter_h - 1) / 2))
        pad_w1 = int(math.floor((filter_w - 1) / 2))
        pad_w2 = int(math.ceil((filter_w - 1) / 2))
        input_tensor = np.pad(input_tensor, ((0, 0), (0, 0), (pad_h1, pad_h2), (pad_w1, pad_w2)), 'constant', constant_values=0)

        batch, channels, height ,width = input_tensor.shape

        self.input_shape = input_tensor.shape
        self.input_channel = channels

        out_h = int((height - filter_h) / self.stri_h) + 1
        out_w = int((width - filter_w) / self.stri_w) + 1

        self.input_col = Tool().im2col(input_tensor, filter_h, filter_w, out_h, out_w)
        self.weight_col = self.weights.reshape(self.kern_num, -1).T

        # add bias to the end of weight_col
        weight_col_bias = np.vstack((self.weight_col, self.bias))

        # add one col 1 to the end of input_col
        bias_input = np.ones((self.input_col.shape[0], 1))
        input_col_plus = np.c_[self.input_col, bias_input]

        output = np.dot(input_col_plus, weight_col_bias)
        output = output.reshape(batch, out_h, out_w, -1).transpose(0, 3, 1, 2)
        self.out_shape = output.shape

        return output


    def backward(self, error_tensor):
        pass

    @property
    def gradient_weights(self):
        return self.gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, value):
        self.gradient_weights = value

    @property
    def gradient_bias(self):
        return self.gradient_bias

    @gradient_bias.setter
    def gradient_bias(self, value):
        self.gradient_bias = value

