# dropout.py
import numpy as np
import math
import random

from lmnn.layers.struct import LayerStruct

class DropoutLayer(LayerStruct):
    def __init__(self, drop_rate=0.2):
        super().__init__()
        self.drop_rate = drop_rate
        self.nb_neurons_kept = None
        self.removed_index = None
        self.Z = None

    def activate(self, previous_layer_act):
        self.nb_neurons_kept = math.floor(previous_layer_act.shape[0] * self.drop_rate)
        self.removed_index = random.sample(range(0, previous_layer_act.shape[0]), self.nb_neurons_kept)

        previous_layer_act[self.removed_index, :] = 0
        return previous_layer_act
    
    def dw(self, m, next_layer_dz, previous_layer_act):
        return 1/m * np.dot(next_layer_dz, previous_layer_act.T)

    def db(self, m, next_layer_dz):
        return 1/m * np.sum(next_layer_dz, axis=1, keepdims=True)

    def da(self, previous_layer_act):
        return previous_layer_act

    def dz(self, next_layer_dz, previous_layer_act):
        return next_layer_dz
    
    def update(self, dw, db, lr):
        return