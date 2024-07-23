import cupy as cp

from lmnn.initializers.struct import InitializerStruct

class RandomInitializer(InitializerStruct):
    def __init__(self, strategy="classic"):
        self.strategy = strategy

    def generate_weights(self, input_dim, output_dim):
        if self.strategy == "small":
            return cp.random.rand(output_dim, input_dim)
        elif self.strategy == "classic":
            return cp.random.randn(output_dim, input_dim)
        else:
            raise ValueError("Unsupported strategy for random initializer")
