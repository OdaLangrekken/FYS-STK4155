import os
import sys

current_path = os.getcwd()
sys.path.append(current_path + '\Project1\project1')

from linear_model import LinearModel
from error_metrics import MSE
from make_data import FrankeFunction
import unittest
import numpy as np
import pandas as pd


class TestLinearModel1(unittest.TestCase):

    def test_linear_model_simple(self):

        # Create a simple dataset to test LinearModel
        x = np.linspace(0, 10, 20).reshape(-1, 1) 
        y = 2 + 3*x

        # Fit LinearModel
        lm = LinearModel()
        lm.fit(x, y)

        # Test that coefficients are almost equal to 2 and 3
        self.assertAlmostEqual(lm.coeffs[0][0], 2)
        self.assertAlmostEqual(lm.coeffs[1][0], 3)

    def test_linear_model_identity_matrix(self):

        # Test that mean squared error is 1 when using identity matrix as design matrix
        X = np.identity(20)
        y = np.random.rand(20)

        lm = LinearModel()
        lm.fit(X, y)

        y_pred = lm.predict(X)
        mse = MSE(y, y_pred)
        print(mse)

        self.assertAlmostEqual(mse, 0)

if __name__ == '__main__':
    unittest.main()