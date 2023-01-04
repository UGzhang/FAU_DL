import math
import numpy as np


class Constant:
    def __init__(self, value=0.1):
        self.__value = value

    def initialize(self,weights_shape, fan_in=None, fan_out=None):
        return self.__value + np.zeros(weights_shape)


class UniformRandom:
    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in=None, fan_out=None):
        return np.random.uniform(0, 1, size=weights_shape)


class Xavier:
    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        # Zero-mean Gaussian
        return np.random.normal(0, math.sqrt(2/(fan_in+fan_out)), size=weights_shape)


class He:
    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in, fan_out=None):
        # Zero-mean Gaussian
        return np.random.normal(0, math.sqrt(2/fan_in), size=weights_shape)