# struct.py
import abc
from abc import ABC, abstractmethod

class ActivationStruct(ABC):

    def __init__(self):
        self._dz = None

    @abstractmethod
    def activate(self, previous_layer_act):
        pass

    @abstractmethod
    def dz(self, weights, next_layer_dz, previous_layer_act):
        pass
