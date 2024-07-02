# dense.py
from .struct import LayerStruct
import numpy as np
from ..activations.struct import ActivationStruct

class DenseLayer(LayerStruct):
    def __init__(self, activation:ActivationStruct, nb_neurons:int=8):
        super().__init__()
        self.nb_neurons = nb_neurons
        self.activation = activation

    def activate(self, previous_layer_act):
        #Set base values in case of first iteration
        if self.weights == None:
            self.weights = np.random.randn(self.nb_neurons, len(previous_layer_act))
        if self.biais == None:
            self.biais = np.random.randn(self.nb_neurons)

        Z = self.weights.dot(previous_layer_act) + self.biais

        return self.activation.activate(Z)
