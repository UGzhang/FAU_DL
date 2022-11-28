class BaseLayer:
    def __init__(self):
        self.trainable = False

    def forward(self, input_tensor):
        return 0

    def backward(self, error_tensor):
        return 0
