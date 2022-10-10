from project1_code.linear_model import LinearModel
from project1_code.make_and_prepare_data import FrankeFunction, create_design_matrix
from project1_code.model_selection import *
import unittest
import numpy as np
import pandas as pd

class TestResamplingMethods(unittest.TestCase):

    def test_bootstrap_trainvstest_indices(self):

        # Test that test set only contains the data not sampled
        print('-------------------------------------------------------------------')
        print('-------------------------------------------------------------------')
        print('Bootstrap: Test that test set only contains the data not sampled')
        print('-------------------------------------------------------------------')
        # Make data
        x = np.random.uniform(0, 1, 20)
        y = np.random.uniform(0, 1, 20)
        z = FrankeFunction(x, y)
        X = create_design_matrix(x, y, 5)

        X_sample, z_sample, X_test, z_test = make_bootstrap_sample(X, z)
        
        sample_rows = X_sample.index.unique().sort_values().tolist()
        test_rows = X_test.index.sort_values().tolist()

        print(f'Rows selected for sample: {sample_rows}')
        print(f'Rows selected for test: {test_rows}')

        for i in range(len(test_rows)):
            self.assertFalse(test_rows[i] in sample_rows, 'Some rows are in both sample and test set!')

    def test_bootstrap_same_test_size(self):
        print('-------------------------------------------------------------------')
        print('-------------------------------------------------------------------')
        print('Bootstrap: Test that X_test and z_test have same size')
        print('-------------------------------------------------------------------')
        # Make data
        x = np.random.uniform(0, 1, 20)
        y = np.random.uniform(0, 1, 20)
        z = FrankeFunction(x, y)
        X = create_design_matrix(x, y, 5)

        X_sample, z_sample, X_test, z_test = make_bootstrap_sample(X, z)
        
        print(f'Size of X_test: {len(X_test)}')
        print(f'Size of z_test: {len(z_test)}')

        self.assertEqual(len(X_test), len(z_test), 'X_test and z_test are not the same size!')

if __name__ == '__main__':
    
    unittest.main()
