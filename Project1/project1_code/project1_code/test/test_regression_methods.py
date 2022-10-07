from project1_code.regression_methods import OLS, ridge
from project1_code.model_selection import MSE
import unittest
import numpy as np
import pandas as pd


class TestRegressionMethods(unittest.TestCase):

    def test_OLS_identity_matrix(self):
        print('-------------------------------------------------------------------')
        print('-------------------------------------------------------------------')
        print('Testing that the mean squared error is 0 when using OLS and identity matrix as design matrix')
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
