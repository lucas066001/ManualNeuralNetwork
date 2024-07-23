# dense.py
import numpy as np

from lmnn.activations.struct import ActivationStruct

class SigmoidActivation(ActivationStruct):
    def __init__(self):
        super().__init__()

    def activate(self, Z):
        sig = 1 / (1 + np.exp(-Z))
        return sig
    
    def da(self, previous_layer_act):
        return previous_layer_act * (1 - previous_layer_act)