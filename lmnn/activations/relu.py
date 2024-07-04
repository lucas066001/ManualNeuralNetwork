# dense.py
from .struct import ActivationStruct
import numpy as np

class ReluActivation(ActivationStruct):
    def __init__(self):
        super().__init__()

    def activate(self, Z):
        return np.maximum(0, Z)
    
    def dz(self, weights, next_layer_dz, previous_layer_act, current_layer_Z):
        previous_layer_act[previous_layer_act <= 0] = 0
        previous_layer_act[previous_layer_act > 0] = 1

        test = np.dot(weights.T, next_layer_dz)
        #print(weights.shape)
        #print(next_layer_dz.shape)
        #print(next_layer_dz.shape)
        #print(test.shape)
        #print(np.unique(np.dot(weights.T, next_layer_dz) * previous_layer_act, return_counts=True))
        return np.dot(weights.T, next_layer_dz) * previous_layer_act