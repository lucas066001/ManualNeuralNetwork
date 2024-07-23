import unittest
import numpy as np

# Import should be able to retreive all activations more conviniently
from lmnn.layers.structures import DenseLayer, DropoutLayer, OutputLayer, LayerStruct
from lmnn.activations.functions import SigmoidActivation

from unittest.mock import Mock, patch, call
from lmnn.layers.dense import DenseLayer
from lmnn.activations.struct import ActivationStruct
from lmnn.initializers.struct import InitializerStruct

class TestDenseLayer(unittest.TestCase):

    def setUp(self):
        self.mock_activation = Mock(spec=ActivationStruct)
        self.mock_initializer = Mock(spec=InitializerStruct)
        self.dense_layer = DenseLayer(activation=self.mock_activation, initializer=self.mock_initializer, nb_neurons=3)

    def test_initialization(self):
        self.assertEqual(self.dense_layer.nb_neurons, 3)
        self.assertEqual(self.dense_layer.activation, self.mock_activation)
        self.assertEqual(self.dense_layer.initializer, self.mock_initializer)
        self.assertIsNone(self.dense_layer.Z)
        self.assertIsNone(self.dense_layer.weights)
        self.assertIsNone(self.dense_layer.biais)

    def test_activate(self):
        previous_layer_act = np.array([[0.1], [0.2]])
        weights = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        biais = np.array([[0.1], [0.2], [0.3]])
        self.dense_layer.biais = biais
        Z = np.dot(weights, previous_layer_act) + biais
        activation_output = np.array([[0.5], [0.6], [0.7]])

        self.mock_initializer.generate_weights.return_value = weights
        self.mock_activation.activate.return_value = activation_output

        result = self.dense_layer.activate(previous_layer_act)

        self.assertTrue((self.dense_layer.weights == weights).all())
        self.assertEqual(self.dense_layer.biais.shape, (3, 1))

        np.testing.assert_almost_equal(self.dense_layer.Z, Z)
        np.testing.assert_almost_equal(result, activation_output)

    def test_dw(self):
        m = 4
        next_layer_dz = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.9, 1.0, 1.1, 1.2]])
        previous_layer_act = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])
        expected_dw = 1/m * np.dot(next_layer_dz, previous_layer_act.T)

        dw = self.dense_layer.dw(m, next_layer_dz, previous_layer_act)
        
        np.testing.assert_almost_equal(dw, expected_dw)

    def test_db(self):
        m = 4
        next_layer_dz = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.9, 1.0, 1.1, 1.2]])
        expected_db = 1/m * np.sum(next_layer_dz, axis=1, keepdims=True)

        db = self.dense_layer.db(m, next_layer_dz)
        
        np.testing.assert_almost_equal(db, expected_db)

    def test_dz(self):
        next_layer_dz = np.array([[0.1], [0.2], [0.3]])
        previous_layer_act = np.array([[0.1], [0.2]])
        weights = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])

        self.mock_initializer.generate_weights.return_value = weights
        self.dense_layer.activate(previous_layer_act)
        
        expected_dz = np.dot(weights.T, next_layer_dz)

        dz = self.dense_layer.dz(next_layer_dz, previous_layer_act)
        
        np.testing.assert_almost_equal(dz, expected_dz)

    def test_da(self):
        previous_layer_act = np.array([[0.5], [0.6], [0.7]])
        da_output = np.array([[0.25], [0.24], [0.21]])

        self.mock_activation.da.return_value = da_output

        da = self.dense_layer.da(previous_layer_act)

        self.mock_activation.da.assert_called_once_with(previous_layer_act)
        np.testing.assert_almost_equal(da, da_output)

    def test_update(self):
        self.dense_layer.weights = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        self.dense_layer.biais = np.array([[0.1], [0.2], [0.3]])
        dw = np.array([[0.01, 0.02], [0.03, 0.04], [0.05, 0.06]])
        db = np.array([[0.01], [0.02], [0.03]])
        lr = 0.1

        self.dense_layer.update(dw, db, lr)

        expected_weights = np.array([[0.099, 0.198], [0.297, 0.396], [0.495, 0.594]])
        expected_biais = np.array([[0.099], [0.198], [0.297]])

        np.testing.assert_almost_equal(self.dense_layer.weights, expected_weights)
        np.testing.assert_almost_equal(self.dense_layer.biais, expected_biais)

class TestDropoutLayer(unittest.TestCase):

    def setUp(self):
        self.dropout_layer = DropoutLayer(drop_rate=0.5)

    @patch('random.sample')
    def test_activate(self, mock_sample):
        previous_layer_act = np.array([[0.1], [0.2], [0.3], [0.4]])
        mock_sample.return_value = [1, 3]

        result = self.dropout_layer.activate(previous_layer_act.copy())

        expected_output = np.array([[0.1], [0.0], [0.3], [0.0]])
        
        np.testing.assert_almost_equal(result, expected_output)
        self.assertEqual(self.dropout_layer.nb_neurons_kept, 2)
        self.assertEqual(self.dropout_layer.removed_index, [1, 3])

    @patch('random.sample')
    def test_dw(self, mock_sample):
        m = 4
        next_layer_dz = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]])
        previous_layer_act = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]])
        mock_sample.return_value = [1, 3]
        
        self.dropout_layer.activate(previous_layer_act.copy())

        previous_layer_act[self.dropout_layer.removed_index, :] = 0

        expected_dw = 1/m * np.dot(next_layer_dz, previous_layer_act.T)

        dw = self.dropout_layer.dw(m, next_layer_dz, previous_layer_act.copy())

        np.testing.assert_almost_equal(dw, expected_dw)

    def test_db(self):
        m = 4
        next_layer_dz = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]])
        expected_db = 1/m * np.sum(next_layer_dz, axis=1, keepdims=True)

        db = self.dropout_layer.db(m, next_layer_dz)

        np.testing.assert_almost_equal(db, expected_db)

    @patch('random.sample')
    def test_da(self, mock_sample):
        previous_layer_act = np.array([[0.1], [0.2], [0.3], [0.4]])
        mock_sample.return_value = [1, 3]

        self.dropout_layer.activate(previous_layer_act.copy())
        expected_da = previous_layer_act.copy()

        da = self.dropout_layer.da(previous_layer_act.copy())

        np.testing.assert_almost_equal(da, expected_da)

    @patch('random.sample')
    def test_dz(self, mock_sample):
        next_layer_dz = np.array([[0.1], [0.2], [0.3], [0.4]])
        previous_layer_act = np.array([[0.1], [0.2], [0.3], [0.4]])
        mock_sample.return_value = [1, 3]

        self.dropout_layer.activate(previous_layer_act.copy())
        expected_dz = next_layer_dz.copy()

        dz = self.dropout_layer.dz(next_layer_dz.copy(), previous_layer_act.copy())

        np.testing.assert_almost_equal(dz, expected_dz)

class TestOutputLayer(unittest.TestCase):

    def setUp(self):
        self.mock_activation = Mock(spec=ActivationStruct)
        self.mock_initializer = Mock(spec=InitializerStruct)
        self.output_layer = OutputLayer(activation=self.mock_activation, initializer=self.mock_initializer, nb_classes=3)

    def test_initialization(self):
        self.assertEqual(self.output_layer.nb_neurons, 3)
        self.assertEqual(self.output_layer.activation, self.mock_activation)
        self.assertEqual(self.output_layer.initializer, self.mock_initializer)
        self.assertIsNone(self.output_layer.Z)
        self.assertIsNone(self.output_layer.weights)
        self.assertIsNone(self.output_layer.biais)

    def test_activate(self):
        previous_layer_act = np.array([[0.1], [0.2]])
        weights = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        biais = np.array([[0.1], [0.2], [0.3]])
        self.output_layer.biais = biais
        Z = np.dot(weights, previous_layer_act) + biais
        activation_output = np.array([[0.5], [0.6], [0.7]])

        self.mock_initializer.generate_weights.return_value = weights
        self.mock_activation.activate.return_value = activation_output

        result = self.output_layer.activate(previous_layer_act)

        self.assertTrue((self.output_layer.weights == weights).all())
        self.assertEqual(self.output_layer.biais.shape, (3, 1))

        np.testing.assert_almost_equal(self.output_layer.Z, Z)
        np.testing.assert_almost_equal(result, activation_output)

    def test_dw(self):
        m = 4
        next_layer_dz = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.9, 1.0, 1.1, 1.2]])
        previous_layer_act = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])
        expected_dw = 1/m * np.dot(next_layer_dz, previous_layer_act.T)

        dw = self.output_layer.dw(m, next_layer_dz, previous_layer_act)
        
        np.testing.assert_almost_equal(dw, expected_dw)

    def test_db(self):
        m = 4
        next_layer_dz = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.9, 1.0, 1.1, 1.2]])
        expected_db = 1/m * np.sum(next_layer_dz, axis=1, keepdims=True)

        db = self.output_layer.db(m, next_layer_dz)
        
        np.testing.assert_almost_equal(db, expected_db)

    def test_dz(self):
        next_layer_dz = np.array([[0.1], [0.2], [0.3]])
        previous_layer_act = np.array([[0.1], [0.2]])
        weights = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])

        self.mock_initializer.generate_weights.return_value = weights
        self.output_layer.activate(previous_layer_act)
        
        expected_dz = np.dot(weights.T, next_layer_dz)

        dz = self.output_layer.dz(next_layer_dz, previous_layer_act)
        
        np.testing.assert_almost_equal(dz, expected_dz)

    def test_da(self):
        previous_layer_act = np.array([[0.5], [0.6], [0.7]])
        da_output = np.array([[0.25], [0.24], [0.21]])

        self.mock_activation.da.return_value = da_output

        da = self.output_layer.da(previous_layer_act)

        self.mock_activation.da.assert_called_once_with(previous_layer_act)
        np.testing.assert_almost_equal(da, da_output)

    def test_update(self):
        self.output_layer.weights = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        self.output_layer.biais = np.array([[0.1], [0.2], [0.3]])
        dw = np.array([[0.01, 0.02], [0.03, 0.04], [0.05, 0.06]])
        db = np.array([[0.01], [0.02], [0.03]])
        lr = 0.1

        self.output_layer.update(dw, db, lr)

        expected_weights = np.array([[0.099, 0.198], [0.297, 0.396], [0.495, 0.594]])
        expected_biais = np.array([[0.099], [0.198], [0.297]])

        np.testing.assert_almost_equal(self.output_layer.weights, expected_weights)
        np.testing.assert_almost_equal(self.output_layer.biais, expected_biais)

if __name__ == '__main__':
    unittest.main()