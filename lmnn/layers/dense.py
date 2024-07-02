# dense.py
from .struct import LayerStruct
import numpy as np

class DenseLayer(LayerStruct):
    def __init__(self, nb_neurons=8, activation="sig"):
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
        
        print("I'm activating !")
