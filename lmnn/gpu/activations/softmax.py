# dense.py
import cupy as cp

from lmnn.activations.struct import ActivationStruct

class SoftMaxActivation(ActivationStruct):
    def __init__(self):
        super().__init__()

    def activate(self, Z):
        exp_logits = cp.exp(Z - cp.max(Z, axis=-1, keepdims=True))
        return exp_logits / cp.sum(exp_logits, axis=-1, keepdims=True)
    
    def da(self, previous_layer_act):
        return cp.ones((previous_layer_act.shape[0], previous_layer_act.shape[1]))