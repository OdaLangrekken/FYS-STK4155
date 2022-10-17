from project1_code.regression_methods import OLS, ridge
from project1_code.model_selection import MSE
from project1_code.make_and_prepare_data import create_design_matrix_1d
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

    def test_ridge_lambda_0(self):
        print('-------------------------------------------------------------------')
        print('-------------------------------------------------------------------')
        print('Test that ridge regression returns same coefficients as OLS when lambda=0')
        print('-------------------------------------------------------------------')

        x = np.linspace(0, 10, 20)
        y = 2 + 5*x + 3*x**2

        X = create_design_matrix_1d(x, 2)

        # Scale data for ridge regression
        X_scaled = X - np.mean(X, axis=0)

        # Add bias 
        X.insert(0, 'bias', np.ones(len(X)))
        X_scaled.insert(0, 'bias', np.ones(len(X_scaled)))

        # FInd OLS coefficients
        coeffs_OLS = OLS(X, y).tolist()
        # Find ridge coefficients

        coeffs_ridge = ridge(X, y, lamb=0).tolist()

        print(f'Coefficients from OLS: {coeffs_OLS}')
        print(f'Coefficients from Ridge: {coeffs_ridge}')
        np.testing.assert_array_equal(coeffs_OLS, coeffs_ridge, 'Ridge regression not returning same result as OLS when lambda=0!')

if __name__ == '__main__':
    unittest.main()
