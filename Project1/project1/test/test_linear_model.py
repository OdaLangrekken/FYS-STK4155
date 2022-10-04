import os
import sys

current_path = os.getcwd()
sys.path.append(current_path + '\Project1\project1')

from linear_model import LinearModel
from model_selection import MSE
import unittest
import numpy as np
import pandas as pd


class TestLinearModel1(unittest.TestCase):

    def test_linear_model_simple(self):

        print('Testing OSL method with a simple function, y=2+5x')
        # Create a simple dataset to test LinearModel
        x = np.linspace(0, 10, 20).reshape(-1, 1) 
        y = 2 + 5*x

        # Fit LinearModel
        lm = LinearModel()
        lm.fit(x, y)

        print(f'Fitted coefficients: {lm.coeffs.T}')
        # Test that coefficients are almost equal to 2 and 3, within 0.0001
        error_message = 'fitted coefficients are not almost equal to true coefficients'
        self.assertAlmostEqual(lm.coeffs[0][0], 2, delta = 0.0001)
        self.assertAlmostEqual(lm.coeffs[1][0], 5, delta = 0.0001)

    def test_linear_model_identity_matrix(self):

        # Test that mean squared error is 1 when using identity matrix as design matrix
        X = np.identity(20)
        y = np.random.rand(20)

        lm = LinearModel()
        lm.fit(X, y)

        y_pred = lm.predict(X)
        mse = MSE(y, y_pred)
        print(mse)

        # Test that MSE is equal to 0, within a 1E-20 precision error
        self.assertAlmostEqual(mse, 0, delta = 1E-20)

if __name__ == '__main__':
    unittest.main()
