# dense.py
from .struct import ActivationStruct
import numpy as np
from sklearn.preprocessing import Normalizer

class SigmoidActivation(ActivationStruct):
    def __init__(self):
        super().__init__()
        self.scaler = Normalizer()

    def activate(self, Z):
        # print("Before Z.min()")
        # print(Z.min())
        # print(Z.max())

        # Z = ((Z - Z.min()) / (Z.max() - Z.min())) * (10 - -10) + (-10)

        # Z = Z / Z.max()
        # print("After Z.min()")
        # print(Z.min())
        # print(Z.max())

        # raise ValueError("STOP")
        return 1 / (1 + np.exp(-Z))
    
    def dz(self, previous_layer_act):
        return previous_layer_act * (1 - previous_layer_act)