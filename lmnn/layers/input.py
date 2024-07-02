# dense.py
from .struct import LayerStruct
import numpy as np
from ..activations.struct import ActivationStruct

class InputLayer(LayerStruct):
    def __init__(self, X:np.ndarray):
        super().__init__()
        self.nb_neurons = X.shape[1]
        self.X = X

    def activate(self, previous_layer_act):
        return self.X
    
    def dw(self, m, next_layer_dz, previous_layer_act):
        raise RuntimeError("Library using missunderstood, you should only use input layer as your first layer. So, you'll never need dw")

    def db(self, m, next_layer_dz):
        raise RuntimeError("Library using missunderstood, you should only use input layer as your first layer. So, you'll never need db")

    def dz(self, next_layer_dz, previous_layer_act):
        raise RuntimeError("Library using missunderstood, you should only use input layer as your first layer. So, you'll never need dz")
    
    def update(self, dw, db, lr):
        raise RuntimeError("Library using missunderstood, you should only use input layer as your first layer. So, you'll never need to update it")
