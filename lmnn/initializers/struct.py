# struct.py
import abc
from abc import ABC, abstractmethod
from ..activations.struct import ActivationStruct
import numpy as np

class InitializerStruct(ABC):

    @abstractmethod
    def generate_weights(self, input_dim, output_dim):
        pass
