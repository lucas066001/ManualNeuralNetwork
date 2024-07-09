# struct.py
import abc
from abc import ABC, abstractmethod

class LossStruct(ABC):

    def __init__(self):
        self._dz = None

    @abstractmethod
    def compute_loss(self, A, y_true):
        pass

    @abstractmethod
    def dl(self, A, y_true):
        pass
