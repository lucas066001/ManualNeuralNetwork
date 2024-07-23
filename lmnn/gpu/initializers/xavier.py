# xavier.py
import cupy as cp

from lmnn.initializers.struct import InitializerStruct

class XavierInitializer(InitializerStruct):
    def __init__(self, strategy="uniform"):
        self.strategy = strategy

    def generate_weights(self, input_dim, output_dim):
        if self.strategy == "uniform":
            limit = cp.sqrt(6 / (input_dim + output_dim))
            return cp.random.uniform(-limit, limit, size=(output_dim, input_dim))
        elif self.strategy == "normal":
            stddev = cp.sqrt(2 / (input_dim + output_dim))
            return cp.random.normal(0, stddev, size=(output_dim, input_dim))
        else :
            raise ValueError("Unsupported strategy for xavier initializer")