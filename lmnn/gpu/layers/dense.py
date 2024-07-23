# dense.py
import cupy as cp

from lmnn.layers.struct import LayerStruct
from lmnn.activations.struct import ActivationStruct
from lmnn.initializers.struct import InitializerStruct

class DenseLayer(LayerStruct):
    def __init__(self, activation: ActivationStruct, initializer: InitializerStruct, nb_neurons: int = 8):
        super().__init__()
        self.nb_neurons = nb_neurons
        self.activation = activation
        self.initializer = initializer
        self.Z = None

    def activate(self, previous_layer_act):
        if self.weights is None:
            self.weights = self.initializer.generate_weights(len(previous_layer_act), self.nb_neurons)
        if self.biais is None:
            self.biais = cp.random.rand(self.nb_neurons, 1)
        
        # Compute Z
        self.Z = cp.dot(self.weights, previous_layer_act) + self.biais
        return self.activation.activate(self.Z)
    
    def dw(self, m, next_layer_dz, previous_layer_act):
        return 1 / m * cp.dot(next_layer_dz, previous_layer_act.T)

    def db(self, m, next_layer_dz):
        return 1 / m * cp.sum(next_layer_dz, axis=1, keepdims=True)

    def dz(self, next_layer_dz, previous_layer_act):
        return cp.dot(self.weights.T, next_layer_dz)
    
    def da(self, previous_layer_act):
        return self.activation.da(previous_layer_act)

    def update(self, dw, db, lr):
        self.weights -= lr * dw
        self.biais -= lr * db
