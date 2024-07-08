# dense.py
from .struct import ActivationStruct
import numpy as np
from sklearn.preprocessing import Normalizer

class SigmoidActivation(ActivationStruct):
    def __init__(self):
        super().__init__()
        self.scaler = Normalizer()

    def activate(self, Z):
        Z = self.scaler.fit_transform(Z)
        return 1 / (1 + np.exp(-Z))
    
    def dz(self, previous_layer_act):
        return previous_layer_act * (1 - previous_layer_act)