# dense.py
import numpy as np

from lmnn.activations.struct import ActivationStruct

class ReluActivation(ActivationStruct):
    def __init__(self):
        super().__init__()

    def activate(self, Z):
        return np.maximum(0, Z)
    
    def da(self, previous_layer_act):
        da = np.where(previous_layer_act < 0, 0, 1)
        return da