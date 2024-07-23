import unittest
import numpy as np
import cupy as cp


# Import should be able to retreive all activations more conviniently
from lmnn.gpu.initializers.functions import HeInitializer, RandomInitializer, XavierInitializer, InitializerStruct


class TestHeInitializer(unittest.TestCase):

    def setUp(self):
        self.initializer = HeInitializer()

    def test_generate_weights_shape(self):
        input_dim = 4
        output_dim = 3
        weights = self.initializer.generate_weights(input_dim, output_dim)
        self.assertEqual(weights.shape, (output_dim, input_dim))

    def test_generate_weights_distribution(self):
        input_dim = 1000
        output_dim = 1000
        weights = self.initializer.generate_weights(input_dim, output_dim)
        stddev = np.sqrt(2 / input_dim)
        
        # Check mean is approximately 0
        self.assertAlmostEqual(np.mean(weights.get()), 0, places=1)
        
        # Check stddev is approximately sqrt(2 / input_dim)
        self.assertAlmostEqual(np.std(weights.get()), stddev, places=1)

class TestRandomInitializer(unittest.TestCase):

    def test_generate_weights_shape_classic(self):
        initializer = RandomInitializer(strategy="classic")
        input_dim = 4
        output_dim = 3
        weights = initializer.generate_weights(input_dim, output_dim)
        self.assertEqual(weights.shape, (output_dim, input_dim))

    def test_generate_weights_shape_small(self):
        initializer = RandomInitializer(strategy="small")
        input_dim = 4
        output_dim = 3
        weights = initializer.generate_weights(input_dim, output_dim)
        self.assertEqual(weights.shape, (output_dim, input_dim))

    def test_generate_weights_distribution_classic(self):
        initializer = RandomInitializer(strategy="classic")
        input_dim = 1000
        output_dim = 1000
        weights = initializer.generate_weights(input_dim, output_dim)
        
        # Check mean is approximately 0 for normal distribution
        self.assertAlmostEqual(np.mean(weights.get()), 0, places=1)
        
        # Check stddev is approximately 1 for normal distribution
        self.assertAlmostEqual(np.std(weights.get()), 1, places=1)

    def test_generate_weights_distribution_small(self):
        initializer = RandomInitializer(strategy="small")
        input_dim = 1000
        output_dim = 1000
        weights = initializer.generate_weights(input_dim, output_dim)
        
        # Check mean is approximately 0.5 for uniform distribution
        self.assertAlmostEqual(np.mean(weights.get()), 0.5, places=1)
        
        # Check stddev is approximately sqrt(1/12) for uniform distribution
        expected_stddev = np.sqrt(1 / 12)
        self.assertAlmostEqual(np.std(weights.get()), expected_stddev, places=1)

    def test_unsupported_strategy(self):
        initializer = RandomInitializer(strategy="unsupported")
        with self.assertRaises(ValueError):
            initializer.generate_weights(4, 3)

class TestXavierInitializer(unittest.TestCase):

    def test_generate_weights_shape_uniform(self):
        initializer = XavierInitializer(strategy="uniform")
        input_dim = 4
        output_dim = 3
        weights = initializer.generate_weights(input_dim, output_dim)
        self.assertEqual(weights.shape, (output_dim, input_dim))

    def test_generate_weights_shape_normal(self):
        initializer = XavierInitializer(strategy="normal")
        input_dim = 4
        output_dim = 3
        weights = initializer.generate_weights(input_dim, output_dim)
        self.assertEqual(weights.shape, (output_dim, input_dim))

    def test_generate_weights_distribution_uniform(self):
        initializer = XavierInitializer(strategy="uniform")
        input_dim = 1000
        output_dim = 1000
        weights = initializer.generate_weights(input_dim, output_dim)
        
        limit = np.sqrt(6 / (input_dim + output_dim))
        
        # Check mean is approximately 0
        self.assertAlmostEqual(np.mean(weights.get()), 0, places=1)
        
        # Check the values are within the limits
        self.assertTrue(np.all(weights.get() <= limit))
        self.assertTrue(np.all(weights.get() >= -limit))

    def test_generate_weights_distribution_normal(self):
        initializer = XavierInitializer(strategy="normal")
        input_dim = 1000
        output_dim = 1000
        weights = initializer.generate_weights(input_dim, output_dim)
        
        stddev = np.sqrt(2 / (input_dim + output_dim))
        
        # Check mean is approximately 0
        self.assertAlmostEqual(np.mean(weights.get()), 0, places=1)
        
        # Check stddev is approximately sqrt(2 / (input_dim + output_dim))
        self.assertAlmostEqual(np.std(weights.get()), stddev, places=1)

    def test_unsupported_strategy(self):
        initializer = XavierInitializer(strategy="unsupported")
        with self.assertRaises(ValueError):
            initializer.generate_weights(4, 3)

if __name__ == '__main__':
    unittest.main()