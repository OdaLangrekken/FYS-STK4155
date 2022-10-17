from project2_code.linear_model import LinearModel
from project2_code.regression_methods import OLS, ridge
from project2_code.model_selection import MSE
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
        
if __name__ == '__main__':
    unittest.main()
