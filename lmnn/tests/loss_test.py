import unittest
import numpy as np

# Import should be able to retreive all activations more conviniently
from lmnn.loss.functions import BceLoss, LossStruct

class TestBceLoss(unittest.TestCase):

    def setUp(self):
        self.bce_loss = BceLoss()
        self.epsilon = 1.0e-9
        self.threshold = 1.0e-5

    def test_compute_loss(self):
        A = np.array([[0.9, 0.1, 0.5]])
        y_true = np.array([[1, 0, 1]])
        expected_loss = -(y_true * np.log(np.maximum(self.threshold, A)) + (1 - y_true) * np.log(np.maximum(self.threshold, 1 - A)))
        
        loss = self.bce_loss.compute_loss(A, y_true)
        
        np.testing.assert_almost_equal(loss, expected_loss)

    def test_compute_loss_with_threshold(self):
        A = np.array([[0.1, 0.9, 0.5]])
        y_true = np.array([[0.9, 0.1, 0.95]])
        A = np.maximum(self.threshold, A)
        expected_loss = -(y_true * np.log(A) + (1 - y_true) * np.log(1 - A))
        
        loss = self.bce_loss.compute_loss(A, y_true)
        
        np.testing.assert_almost_equal(loss, expected_loss)

    def test_dl(self):
        A = np.array([[0.9, 0.1, 0.5]])
        y_true = np.array([[1, 0, 1]])
        expected_dl = (A - y_true) / ((A * (1 - A)) + self.epsilon)
        
        dl = self.bce_loss.dl(A, y_true)
        
        np.testing.assert_almost_equal(dl, expected_dl)

    def test_dl_with_epsilon(self):
        A = np.array([[0.9, 0.1, 0.5]])
        y_true = np.array([[1, 0, 1]])
        expected_dl = (A - y_true) / ((A * (1 - A)) + self.epsilon)
        
        dl = self.bce_loss.dl(A, y_true)
        
        np.testing.assert_almost_equal(dl, expected_dl)

if __name__ == '__main__':
    unittest.main()