from project1_code.linear_model import LinearModel
from project1_code.make_and_prepare_data import FrankeFunction, create_design_matrix
from project1_code.model_selection import *
import unittest
import numpy as np
import pandas as pd

class TestResamplingMethods(unittest.TestCase):

    def test_bootstrap_same_sample_size(self):
        print('-------------------------------------------------------------------')
        print('-------------------------------------------------------------------')
        print('Bootstrap: Test that X_samle and z_sample have same size')
        print('-------------------------------------------------------------------')
        # Make data
        x = np.random.uniform(0, 1, 20)
        y = np.random.uniform(0, 1, 20)
        z = FrankeFunction(x, y)
        X = create_design_matrix(x, y, 5)

        X_sample, z_sample = make_bootstrap_sample(X, z)
        
        print(f'Size of X_sample: {len(X_sample)}')
        print(f'Size of z_sample: {len(z_sample)}')

        self.assertEqual(len(X_sample), len(z_sample), 'X_test and z_test are not the same size!')

if __name__ == '__main__':
    
    unittest.main()
