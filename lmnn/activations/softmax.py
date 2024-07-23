# dense.py
import numpy as np

from lmnn.activations.struct import ActivationStruct

class SoftMaxActivation(ActivationStruct):
    def __init__(self):
        super().__init__()

    def activate(self, Z):
        exp_logits = np.exp(Z - np.max(Z, axis=-1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    
    def da(self, previous_layer_act):
        return np.ones((previous_layer_act.shape[0], previous_layer_act.shape[1]))