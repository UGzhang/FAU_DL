import numpy as np
import math, copy
from Layers.Base import BaseLayer, Tool


class Conv(BaseLayer):

    def __init__(self, stride_shape, convolution_shape, num_kernels):
        super().__init__()
        self.trainable = True

        if len(convolution_shape) == 2:
            convolution_shape = convolution_shape + (1,)
            self._Conv1D = True
        else:
            self._Conv1D = False

        self._num_ker = num_kernels
        self._channel = convolution_shape[0]
        self._f_h = convolution_shape[1]
        self._f_w = convolution_shape[2]
        self._s_h = stride_shape[0]
        self._s_w = stride_shape[1] if type(stride_shape) == tuple else 1
        self._input_shape = None
        self._optimizer = None
        self._img_col = None
        self._weight_col = None

        self.weights = np.random.uniform(size=(num_kernels, *convolution_shape))  # 0-1
        self.bias = np.random.uniform(size=num_kernels)

        self.gradient_weights = None
        self.gradient_bias = None


    '''
        input_tensor: [batch,channel,height,width]
    '''
    def forward(self, input_pad):

        if self._Conv1D: input_pad = np.expand_dims(input_pad, axis=3)

        self._input_shape = input_pad.shape

        # SAME padding
        pad_h1 = int(math.floor((self._f_h - 1) / 2))
        pad_h2 = int(math.ceil((self._f_h - 1) / 2))
        pad_w1 = int(math.floor((self._f_w - 1) / 2))
        pad_w2 = int(math.ceil((self._f_w - 1) / 2))

        input_pad = np.pad(input_pad, ((0, 0), (0, 0), (pad_h1, pad_h2), (pad_w1, pad_w2)), 'constant',
                           constant_values=0)

        N, _, H, W = input_pad.shape

        out_h = int((H - self._f_h) / self._s_h) + 1
        out_w = int((W - self._f_w) / self._s_w) + 1

        self._img_col = Tool().im2col(input_pad, [self._f_h, self._f_w], [self._s_h, self._s_w])
        self._weight_col = self.weights.reshape(self._num_ker, -1).T

        out = np.dot(self._img_col, self._weight_col) + self.bias

        output = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        if self._Conv1D: output = np.squeeze(output, axis=3)

        return output

    def backward(self, error_tensor):

        if self._Conv1D: error_tensor = np.expand_dims(error_tensor, axis=3)

        error_tensor = error_tensor.transpose(0, 2, 3, 1).reshape(-1, self._num_ker)

        self.gradient_weights = np.dot(self._img_col.T, error_tensor)
        self.gradient_weights = self.gradient_weights.transpose(1, 0).reshape(self.weights.shape)

        self.gradient_bias = np.sum(error_tensor, axis=0)  # add every row

        err_col = np.dot(error_tensor, self._weight_col.T)

        grad_err = Tool().col2im(err_col, self._input_shape, [self._f_h, self._f_w], [self._s_h, self._s_w])
        if self._Conv1D: grad_err = np.squeeze(grad_err, axis=3)

        if self._optimizer is not None:
            self.weights = self._optimizer.weights.calculate_update(self.weights, self.gradient_weights)
            self.bias = self._optimizer.bias.calculate_update(self.bias, self.gradient_bias)
        return grad_err

    def initialize(self, weights_initializer, bias_initializer):
        self.weights = weights_initializer.initialize(self.weights.shape, np.prod(self._channel * self._f_h * self._f_w),
                                                      np.prod(self._f_h * self._f_w) * self._num_ker)
        self.bias = bias_initializer.initialize(self.bias.shape, 1, self._num_ker)

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, _opt):
        self._optimizer = _opt
        self._optimizer.weights = copy.deepcopy(_opt)
        self._optimizer.bias = copy.deepcopy(_opt)

