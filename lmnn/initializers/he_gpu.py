# he.py
from .struct import InitializerStruct
import cupy as cp

class HeInitializer(InitializerStruct):
    def __init__(self, startegy="uniform"):
        self.strategy = startegy

    def generate_weights(self, input_dim, output_dim):
        stddev = cp.sqrt(2 / input_dim)
        return cp.random.normal(0, stddev, size=(output_dim, input_dim))