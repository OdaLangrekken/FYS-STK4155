import os
import sys

current_path = os.getcwd()
sys.path.append(current_path + '\Project1\project1')

from linear_model import LinearModel
from regression_methods import OLS, ridge
from model_selection import MSE
import unittest
import numpy as np
import pandas as pd


class TestLinearModel(unittest.TestCase):

    def test_linear_model_simple(self):
        print('-------------------------------------------------------------------')
        print('-------------------------------------------------------------------')
        print('Testing OSL method with a simple function, y=2+5x')
        print('-------------------------------------------------------------------')
        # Create a simple dataset to test LinearModel
        x = np.linspace(0, 10, 20).reshape(-1, 1) 
        y = 2 + 5*x

        # Fit LinearModel
        lm = LinearModel()
        lm.fit(x, y)

        print(f'OUTPUT: Fitted coefficients: {lm.coeffs.T}')
        # Test that coefficients are almost equal to 2 and 3, within 0.0001
        error_message = 'fitted coefficients are not almost equal to true coefficients'
        self.assertAlmostEqual(lm.coeffs[0][0], 2, delta = 0.0001)
        self.assertAlmostEqual(lm.coeffs[1][0], 5, delta = 0.0001)

    def test_OLS_identity_matrix(self):
        print('-------------------------------------------------------------------')
        print('-------------------------------------------------------------------')
        print('Testing that the mean squared error is 0 when using identity matrix as design matrix')
        print('-------------------------------------------------------------------')
        # Test that mean squared error is 0 when using identity matrix as design matrix
        X = np.identity(20)
        y = np.random.rand(20)

        coeffs = OLS(X, y)

        y_pred = X @ coeffs
        mse = MSE(y, y_pred)
        print(f'OUTPUT MSE when using identity matrix as input is: {mse}')

        # Test that MSE is equal to 0, within a 1E-20 precision error
        self.assertAlmostEqual(mse, 0, delta = 1E-20)

if __name__ == '__main__':
    unittest.main()
