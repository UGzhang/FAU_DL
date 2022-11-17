import numpy as np


class CrossEntropyLoss:
    def __init__(self):
        self._y = None

    '''
    loss = -∑ln(y_hat + ε)
    '''
    def forward(self, prediction_tensor, label_tensor):
        self._y = prediction_tensor
        return -np.sum(label_tensor * np.log(self._y + np.finfo(float).eps))

    '''
    En-1 = -y / (y_hat + ε)
    '''
    def backward(self, label_tensor):
        return - (label_tensor / (self._y + np.finfo(float).eps))