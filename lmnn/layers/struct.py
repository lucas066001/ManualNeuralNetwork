# struct.py
import abc
from abc import ABC, abstractmethod

class LayerStruct(ABC):
    def __init__(self):
        self._nb_neurons = None
        self._activation = None
        self._weights = None
        self._biais = None

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
    def weights(self):
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
    def biais(self):
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
