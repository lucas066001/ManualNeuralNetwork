# random.py
from .struct import InitializerStruct
import numpy as np

class RandomInitializer(InitializerStruct):
    def __init__(self, startegy="classic"):
        self.strategy = startegy

    def generate_weights(self, input_dim, output_dim):
        if self.strategy == "small": 
            return np.random.rand(output_dim, input_dim)
        elif self.strategy == "classic":
            return np.random.randn(output_dim, input_dim)
        else:
            raise ValueError("Unsupported strategy for random initializer")