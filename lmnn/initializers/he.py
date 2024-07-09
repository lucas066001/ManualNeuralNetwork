# he.py
from .struct import InitializerStruct
import numpy as np

class HeInitializer(InitializerStruct):
    def __init__(self, startegy="uniform"):
        self.strategy = startegy

    def generate_weights(self, input_dim, output_dim):
        stddev = np.sqrt(2 / input_dim)
        return np.random.normal(0, stddev, size=(output_dim, input_dim))