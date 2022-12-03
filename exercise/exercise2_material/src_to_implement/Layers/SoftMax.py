from Layers.Base import BaseLayer
import numpy as np


class SoftMax(BaseLayer):
    def __init__(self):
        super().__init__()
        self.y_hat = None

    '''
    input_tensor:
        rows --> batch size
        cols --> the number of classes
    
    y_hat = exp(Xk) / ∑exp(Xj)
    '''
    def forward(self, input_tensor):
        # minus the max number in a row
        input_tensor -= np.max(input_tensor, axis=1, keepdims=True)
        self.y_hat = np.exp(input_tensor) / np.sum(np.exp(input_tensor), axis=1, keepdims=True)
        return self.y_hat

    '''
    En-1 = y_hat * (En - ∑(y_hat*En))
    '''
    def backward(self, error_tensor):
        cols = error_tensor.shape[1]
        E_sum = np.sum(self.y_hat * error_tensor, axis=1).repeat(cols).reshape((-1, cols))
        return self.y_hat * (error_tensor - E_sum)

