# dense.py
from .struct import ActivationStruct
import numpy as np

class SigmoidActivation(ActivationStruct):
    def __init__(self):
        super().__init__()

    def activate(self, Z):
        return 1 / (1 + np.exp(-Z))
