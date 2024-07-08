# dense.py
from .struct import ActivationStruct
import numpy as np

class ReluActivation(ActivationStruct):
    def __init__(self):
        super().__init__()

    def activate(self, Z):
        Z[Z <= 0 ] = 0
        return Z
    
    def dz(self, previous_layer_act):
        previous_layer_act[previous_layer_act <= 0] = 0
        previous_layer_act[previous_layer_act > 0] = 1
        return previous_layer_act