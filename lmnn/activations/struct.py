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
    def dz(self, previous_layer_act):
        pass
