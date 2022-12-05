import numpy as np
class BaseLayer:
    def __init__(self):
        self.trainable = False

    def forward(self, input_tensor):
        return 0

    def backward(self, error_tensor):
        return 0


class Tool:
    def __init__(self):
        pass

    '''
        input_tensor: 
            image
            4-D [batch,channel,height,width]
        filter: 
            filter size
            2-D [height,width]
        stride:
            stride size
            2-D [height,width]
    '''
    def im2col(self, input_tensor, filter, stride):
        # input_tensor.shape = (N, C, H, W)
        N, C, H, W = input_tensor.shape
        f_h, f_w = filter[0], filter[1]
        s_h, s_w = stride[0], stride[1]

        out_h = (H - f_h) // s_h + 1
        out_w = (W - f_w) // s_w + 1

        col = np.zeros((N, C, f_h, f_w, out_h, out_w))

        for y in range(f_h):
            y_max = y + s_h * out_h
            for x in range(f_w):
                x_max = x + s_w * out_w
                col[:, :, y, x, :, :] = input_tensor[:, :, y:y_max:s_h, x:x_max:s_w]
        col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
        return col

    '''
        col: 
            from method 'im2col'
            4-D [batch,channel,height,width]
        input_shape: 
            4-D [batch,channel,height,width]
        filter: 
            filter size
            2-D [height,width]
        stride:
            stride size
            2-D [height,width]
    '''
    def col2im(self, col, input_shape, filter, stride):

        N, C, H, W = input_shape
        f_h, f_w = filter[0], filter[1]
        s_h, s_w = stride[0], stride[1]

        out_h = (H - f_w) // s_h + 1
        out_w = (W - f_h) // s_w + 1

        # print("reshape = ", N, ",",out_h, ",",out_w, ",",C, ",",filter_h, ",",filter_w)

        col = col.reshape(N, out_h, out_w, C, f_h, f_w).transpose(0, 3, 4, 5, 1, 2)

        # add the padding to image
        img = np.zeros((N, C, H, W))

        for y in range(f_h):
            y_max = y + s_h * out_h
            for x in range(f_w):
                x_max = x + s_w * out_w
                img[:, :, y:y_max:s_h, x:x_max:s_w] += col[:, :, y, x, :, :]

        return img


