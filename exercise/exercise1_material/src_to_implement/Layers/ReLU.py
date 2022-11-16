from Layers.Base import BaseLayer


class ReLU(BaseLayer):

    def __init__(self):
        super().__init__()
        self._x = None

    '''
    x = x > 0 ? x : 0
    '''
    def forward(self, input_tensor):
        self._x = 1 * (input_tensor > 0) * input_tensor
        return self._x

    '''
    En = x > 0 ? En : 0
    '''
    def backward(self, error_tensor):
        return 1 * (self._x > 0) * error_tensor
