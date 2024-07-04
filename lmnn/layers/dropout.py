# dropout.py
from .struct import LayerStruct
import numpy as np
from ..activations.struct import ActivationStruct
import math
import random

class DropoutLayer(LayerStruct):
    def __init__(self, activation:ActivationStruct, nb_neurons:int=8, drop_rate=0.2):
        super().__init__()
        self.nb_neurons = nb_neurons
        self.activation = activation
        self.drop_rate = drop_rate
        self.removed_index = None
        self.Z = None

    def activate(self, previous_layer_act):
        if self.removed_index is None: 
            nb_neurons_kept = math.floor(previous_layer_act.shape[0] * self.drop_rate)
            self.removed_index = random.sample(range(0, previous_layer_act.shape[0]), nb_neurons_kept)
        if self.weights is None:
            self.weights = np.random.randn(self.nb_neurons, len(previous_layer_act))
        if self.biais is None:
            self.biais = np.random.randn(self.nb_neurons, 1)
        
        previous_layer_act[self.removed_index, :] = 0
        self.Z = self.weights.dot(previous_layer_act) + self.biais

        return self.activation.activate(self.Z)
    
    def dw(self, m, next_layer_dz, previous_layer_act):
        return 1/m * np.dot(next_layer_dz, previous_layer_act.T)

    def db(self, m, next_layer_dz):
        return 1/m * np.sum(next_layer_dz, axis=1, keepdims=True)

    def dz(self, next_layer_dz, previous_layer_act):
        return self.activation.dz(self.weights, next_layer_dz, previous_layer_act, self.Z)
    
    def update(self, dw, db, lr):
        self.weights = self.weights - lr * dw
        self.biais = self.biais - lr * db