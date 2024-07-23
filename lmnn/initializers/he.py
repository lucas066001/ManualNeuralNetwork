# he.py
import numpy as np

from lmnn.initializers.struct import InitializerStruct

class HeInitializer(InitializerStruct):

    def generate_weights(self, input_dim, output_dim):
        stddev = np.sqrt(2 / input_dim)
        return np.random.normal(0, stddev, size=(output_dim, input_dim))