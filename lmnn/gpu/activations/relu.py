import cupy as cp

from lmnn.activations.struct import ActivationStruct

class ReluActivation(ActivationStruct):
    def __init__(self):
        super().__init__()

    def activate(self, Z):
        # Utiliser CuPy pour la fonction ReLU sur le GPU
        return cp.maximum(0, Z)
    
    def da(self, previous_layer_act):
        # Utiliser CuPy pour la dérivée de ReLU sur le GPU
        return cp.where(previous_layer_act < 0, 0, 1)
