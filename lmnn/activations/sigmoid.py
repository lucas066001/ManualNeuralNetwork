# dense.py
from .struct import ActivationStruct
import numpy as np

class SigmoidActivation(ActivationStruct):
    def __init__(self):
        super().__init__()

    def activate(self, Z):
        return 1 / (1 + np.exp(-Z))
    
    def dz(self, weights, next_layer_dz, previous_layer_act):
        #print(weights.shape)
        #print(next_layer_dz.shape)
        #print(previous_layer_act.shape)
        return np.dot(weights.T, next_layer_dz) * previous_layer_act * (1 - previous_layer_act)