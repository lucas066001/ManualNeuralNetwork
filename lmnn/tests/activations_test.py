import unittest
import numpy as np

# Import should be able to retreive all activations more conviniently
from lmnn.activations.functions import ReluActivation, SigmoidActivation, SoftMaxActivation, ActivationStruct

class TestSigmoidActivation(unittest.TestCase):
    
    def setUp(self):
        self.activation = SigmoidActivation()

    def test_activate(self):
        # Test with a scalar value
        Z = 0
        expected_output = 0.5
        np.testing.assert_almost_equal(self.activation.activate(Z), expected_output, decimal=6)
        
        # Test with an array of values
        Z = np.array([-1, 0, 1])
        expected_output = np.array([0.26894142, 0.5, 0.73105858])
        np.testing.assert_almost_equal(self.activation.activate(Z), expected_output, decimal=6)

    def test_da(self):
        # Test with a scalar value
        previous_layer_act = 0.5
        expected_output = 0.25
        np.testing.assert_almost_equal(self.activation.da(previous_layer_act), expected_output, decimal=6)
        
        # Test with an array of values
        previous_layer_act = np.array([0.26894142, 0.5, 0.73105858])
        expected_output = np.array([0.19661193, 0.25, 0.19661193])
        np.testing.assert_almost_equal(self.activation.da(previous_layer_act), expected_output, decimal=6)

class TestReluActivation(unittest.TestCase):
    
    def setUp(self):
        self.activation = ReluActivation()

    def test_activate(self):
        # Test with a scalar value
        Z = -1
        expected_output = 0
        np.testing.assert_equal(self.activation.activate(Z), expected_output)
        
        Z = 0
        expected_output = 0
        np.testing.assert_equal(self.activation.activate(Z), expected_output)

        Z = 1
        expected_output = 1
        np.testing.assert_equal(self.activation.activate(Z), expected_output)
        
        # Test with an array of values
        Z = np.array([-1, 0, 1])
        expected_output = np.array([0, 0, 1])
        np.testing.assert_equal(self.activation.activate(Z), expected_output)

    def test_da(self):
        # Test with a scalar value
        previous_layer_act = -1
        expected_output = 0
        np.testing.assert_equal(self.activation.da(previous_layer_act), expected_output)

        previous_layer_act = 0
        expected_output = 1
        np.testing.assert_equal(self.activation.da(previous_layer_act), expected_output)
        
        previous_layer_act = 1
        expected_output = 1
        np.testing.assert_equal(self.activation.da(previous_layer_act), expected_output)
        
        # Test with an array of values
        previous_layer_act = np.array([-1, 0, 1])
        expected_output = np.array([0, 1, 1])
        np.testing.assert_equal(self.activation.da(previous_layer_act), expected_output)

if __name__ == '__main__':
    unittest.main()