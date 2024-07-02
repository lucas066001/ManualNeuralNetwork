# struct.py
import abc
from abc import ABC, abstractmethod

class ActivationStruct(ABC):

    @abstractmethod
    def activate(self, previous_layer_act):
        pass
