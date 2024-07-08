# dense.py
from .struct import LayerStruct
import numpy as np
from ..activations.struct import ActivationStruct

class DenseLayer(LayerStruct):
    def __init__(self, activation:ActivationStruct, nb_neurons:int=8):
        super().__init__()
        self.nb_neurons = nb_neurons
        self.activation = activation
        self.Z = None

    def activate(self, previous_layer_act):
        if self.weights is None:
            self.weights = np.random.randn(self.nb_neurons, len(previous_layer_act))
        if self.biais is None:
            self.biais = np.random.randn(self.nb_neurons, 1)

        self.Z  = self.weights.dot(previous_layer_act) + self.biais
        return self.activation.activate(self.Z)
    
    def dw(self, m, next_layer_dz, previous_layer_act):
        return 1/m * np.dot(next_layer_dz, previous_layer_act.T)

    def db(self, m, next_layer_dz):
        return 1/m * np.sum(next_layer_dz, axis=1, keepdims=True)

    def dz(self, next_layer_dz, previous_layer_act):
        return self.activation.dz(self.weights, next_layer_dz, previous_layer_act, self.Z)
    
    def update(self, dw, db, lr):
        #print("self.weights.shape")
        #print(self.weights.shape)
        #print(dw.shape)
        self.weights = self.weights - lr * dw
        self.biais = self.biais - lr * db