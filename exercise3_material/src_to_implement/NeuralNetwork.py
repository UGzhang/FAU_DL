import copy


class NeuralNetwork:
    def __init__(self, optimizer, weights_initializer, bias_initializer):
        self.optimizer = optimizer
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None
        self.input_tensor = None
        self.label_tensor = None

        # add in ex2
        self._weights_initializer = copy.deepcopy(weights_initializer)
        self._bias_initializer = copy.deepcopy(bias_initializer)

        self._phase = None

    '''
        1.fully connected
        2.relu
        3.fully connected
        4.softmax
    '''
    def forward(self):
        self.input_tensor, self.label_tensor = self.data_layer.next()
        reg_loss = 0
        for layer in self.layers:
            layer.testing_phase = False
            self.input_tensor = layer.forward(self.input_tensor)
            if self.optimizer.regularizer:
                reg_loss += self.optimizer.regularizer.norm(layer.weights)
        return reg_loss+self.loss_layer.forward(self.input_tensor, self.label_tensor)

    def backward(self):
        error_tensor = self.loss_layer.backward(self.label_tensor)
        for layer in reversed(self.layers):
            error_tensor = layer.backward(error_tensor)
        return error_tensor

    def append_layer(self, layer):
        if layer.trainable:
            # add in ex2
            layer.initialize(self._weights_initializer, self._bias_initializer)
            layer.optimizer = copy.deepcopy(self.optimizer)
        self.layers.append(layer)

    def train(self, iterations):
        self.phase = "train"
        for _ in range(iterations):
            self.loss.append(self.forward())
            self.backward()

    def test(self, input_tensor):
        self.phase = "test"
        for layer in self.layers:
            layer.testing_phase = True
            input_tensor = layer.forward(input_tensor)
        return input_tensor

    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, p):
        self._phase = p

