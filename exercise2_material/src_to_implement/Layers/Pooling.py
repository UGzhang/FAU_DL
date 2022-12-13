import numpy as np
from Layers.Base import BaseLayer, Tool


class Pooling(BaseLayer):

    def __init__(self, stride_shape, pooling_shape):
        super().__init__()
        self._pooling_size = pooling_shape
        self._stride_size = stride_shape
        self._input_shape = None
        self._pos_max = None

    def forward(self, input_tensor):
        self._input_shape = input_tensor.shape
        N, C, H, W = input_tensor.shape
        p_h, p_w = self._pooling_size[0], self._pooling_size[1]
        s_h, s_w = self._stride_size[0], self._stride_size[1]

        out_h = (H - p_h) // s_h + 1
        out_w = (W - p_w) // s_w + 1

        # each row in input_col matrix is a filter area
        input_col = Tool().im2col(input_tensor, self._pooling_size, self._stride_size)
        # dimension [N * out_h * out_w, p_h * p_w * C] --> [N * C * out_h * out_w, p_h * p_w]
        input_col = input_col.reshape(-1, p_h * p_w)

        # get the position of max value
        self._pos_max = np.argmax(input_col, axis=1)

        # find the max value in a row
        output = np.max(input_col, axis=1)

        # new shape (N, C, out_h, out_w)
        output = output.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        return output

    def backward(self, error_tensor):

        error_tensor = error_tensor.transpose(0, 2, 3, 1)

        pool_size = self._pooling_size[0] * self._pooling_size[1]

        err_max = np.zeros((error_tensor.size, pool_size))
        # put the max value back to its position; pos_max.shape = (N*out_h*out_w, 1)
        err_max[np.arange(self._pos_max.size), self._pos_max.flatten()] = error_tensor.flatten()

        # err_max.shape = (N, H, W, C, Pool_h*Pool_w)
        err_max = err_max.reshape(error_tensor.shape + (pool_size,))

        err_col = err_max.reshape(err_max.shape[0] * err_max.shape[1] * err_max.shape[2], -1)
        next_err = Tool().col2im(err_col, self._input_shape, self._pooling_size, self._stride_size, True)

        return next_err