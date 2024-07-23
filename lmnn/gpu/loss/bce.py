# he.py
import cupy as cp

from lmnn.loss.struct import LossStruct

class BceLoss(LossStruct):
    def __init__(self):
        self.epsilon = 1.0e-9
        self.threshold = 1.0e-5

    def compute_loss(self, A, y_true):
        A = cp.maximum(self.threshold, A)
        return -(y_true * cp.log(A) + (1 - y_true) * cp.log(1 - A))

    def dl(self, A, y_true):
        return (A - y_true) / ((A * (1 - A)) + self.epsilon)