# struct.py
import abc
from abc import ABC, abstractmethod
import numpy as np

from lmnn.activations.struct import ActivationStruct

class InitializerStruct(ABC):

    @abstractmethod
    def generate_weights(self, input_dim, output_dim):
        pass
