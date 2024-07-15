import cupy as cp
from .struct import LayerStruct
from ..activations.struct import ActivationStruct
from ..initializers.struct import InitializerStruct

class OutputGpuLayer(LayerStruct):
    def __init__(self, activation: ActivationStruct, initializer: InitializerStruct, nb_classes: int):
        super().__init__()
        self.nb_neurons = nb_classes
        self.initializer = initializer
        self.activation = activation
        self.Z = None

    def activate(self, previous_layer_act):
        if self.weights is None:
            self.weights = self.initializer.generate_weights(len(previous_layer_act), self.nb_neurons)
        if self.biais is None:
            self.biais = cp.random.rand(self.nb_neurons, 1)  # Remplacer np.random.rand par cp.random.rand

        self.Z = cp.dot(self.weights, previous_layer_act) + self.biais  # Remplacer np.dot par cp.dot
        return self.activation.activate(self.Z)
    
    def dw(self, m, next_layer_dz, previous_layer_act):
        return 1 / m * cp.dot(next_layer_dz, previous_layer_act.T)  # Remplacer np.dot par cp.dot

    def db(self, m, next_layer_dz):
        return 1 / m * cp.sum(next_layer_dz, axis=1, keepdims=True)  # Remplacer np.sum par cp.sum

    def dz(self, next_layer_dz, previous_layer_act):
        return cp.dot(self.weights.T, next_layer_dz)  # Remplacer np.dot par cp.dot
    
    def da(self, previous_layer_act):
        return self.activation.da(previous_layer_act)  # Utiliser CuPy si l'activation est aussi compatible GPU

    def update(self, dw, db, lr):
        self.weights = self.weights - lr * dw  # Les opérations sont déjà effectuées sur le GPU
        self.biais = self.biais - lr * db  # Les opérations sont déjà effectuées sur le GPU
