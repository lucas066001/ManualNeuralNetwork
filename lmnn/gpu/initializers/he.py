# he.py
import cupy as cp

from lmnn.initializers.struct import InitializerStruct

class HeInitializer(InitializerStruct):
    def __init__(self, strategy="uniform"):
        self.strategy = strategy

    def generate_weights(self, input_dim, output_dim):
        stddev = cp.sqrt(2 / input_dim)
        return cp.random.normal(0, stddev, size=(output_dim, input_dim))