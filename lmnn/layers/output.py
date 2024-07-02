# dense.py
from .struct import LayerStruct
import numpy as np
from ..activations.struct import ActivationStruct

class OutputLayer(LayerStruct):
    def __init__(self, activation:ActivationStruct, nb_classes:int):
        super().__init__()
        self.nb_neurons = nb_classes
        self.activation = activation

    def activate(self, previous_layer_act):
        #print("---------LayerAct")
        #Set base values in case of first iteration
        if self.weights is None:
            self.weights = np.random.randn(self.nb_neurons, len(previous_layer_act))
        if self.biais is None:
            self.biais = np.random.randn(self.nb_neurons, 1)

        #print(previous_layer_act.shape)
        #print(self.biais)
        Z = self.weights.dot(previous_layer_act) + self.biais
        #print("---------LayerAct")
        return self.activation.activate(Z)
    
    def dw(self, m, next_layer_dz, previous_layer_act):
        return 1/m * np.dot(next_layer_dz, previous_layer_act.T)

    def db(self, m, next_layer_dz):
        return 1/m * np.sum(next_layer_dz, axis=1, keepdims=True)

    def dz(self, next_layer_dz, previous_layer_act):
        return self.activation.dz(self.weights, next_layer_dz, previous_layer_act)
    
    def update(self, dw, db, lr):
        self.weights = self.weights - lr * dw
        self.biais= self.biais - lr * db