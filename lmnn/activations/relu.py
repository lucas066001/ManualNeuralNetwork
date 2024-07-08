# dense.py
from .struct import ActivationStruct
import numpy as np

class ReluActivation(ActivationStruct):
    def __init__(self):
        super().__init__()

    def activate(self, Z):
        return np.maximum(0, Z)
    
    def dz(self, previous_layer_act):
        # previous_layer_act[previous_layer_act <= 0] = 0
        # previous_layer_act[previous_layer_act > 0] = 1
        return np.where(previous_layer_act < 0, 0, 1)