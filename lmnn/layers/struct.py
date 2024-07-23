# struct.py
import abc
from abc import ABC, abstractmethod
import numpy as np

from lmnn.activations.struct import ActivationStruct
from lmnn.initializers.struct import InitializerStruct

class LayerStruct(ABC):
    def __init__(self):
        self._nb_neurons:int = None
        self._activation:ActivationStruct = None
        self._initializer:InitializerStruct = None
        self._weights:np.ndarray = None
        self._biais:np.ndarray = None

    @property
    def nb_neurons(self):
        return self._nb_neurons

    @nb_neurons.setter
    def nb_neurons(self, nb_neurons):
        self._nb_neurons = nb_neurons

    @nb_neurons.getter
    def nb_neurons(self):
        return self._nb_neurons

    @nb_neurons.deleter
    def nb_neurons(self):
        del self._nb_neurons

    @property
    def activation(self):
        return self._activation

    @activation.setter
    def activation(self, activation):
        self._activation = activation

    @activation.getter
    def activation(self):
        return self._activation

    @activation.deleter
    def activation(self):
        del self._activation

    @property
    def initializer(self):
        return self._initializer

    @initializer.setter
    def initializer(self, initializer):
        self._initializer = initializer

    @initializer.getter
    def initializer(self):
        return self._initializer

    @initializer.deleter
    def initializer(self):
        del self._initializer

    @property
    def weights(self) -> np.ndarray:
        return self._weights

    @weights.setter
    def weights(self, weights):
        self._weights = weights

    @weights.getter
    def weights(self):
        return self._weights

    @weights.deleter
    def weights(self):
        del self._weights

    @property
    def biais(self) -> np.ndarray:
        return self._biais

    @biais.setter
    def biais(self, biais):
        self._biais = biais

    @biais.getter
    def biais(self):
        return self._biais

    @biais.deleter
    def biais(self):
        del self._biais

    @abstractmethod
    def activate(self, previous_layer_act):
        pass

    @abstractmethod
    def dw(self, m, next_layer_dz, previous_layer_act):
        pass

    @abstractmethod
    def db(self, m, next_layer_dz):
        pass

    @abstractmethod
    def dz(self, next_layer_dz, previous_layer_act):
        pass

    @abstractmethod
    def da(self, previous_layer_act):
        pass

    @abstractmethod
    def update(self, dw, db, lr):
        pass
