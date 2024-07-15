import cupy as cp
from .struct import ActivationStruct

class SigmoidGpuActivation(ActivationStruct):
    def __init__(self):
        super().__init__()

    def activate(self, Z):
        # Application de la fonction sigmoïde sur le GPU
        sig = 1 / (1 + cp.exp(-Z))
        return sig
    
    def da(self, previous_layer_act):
        # Derivée de la fonction sigmoïde sur le GPU
        return previous_layer_act * (1 - previous_layer_act)
