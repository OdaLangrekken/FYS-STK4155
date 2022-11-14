from project2_code.model_classes import LinearModel
from project2_code.model_selection import MSE, R2
import unittest
import numpy as np
import pandas as pd

class TestErrorMetrics(unittest.TestCase):

    def test_mse_0_when_spoton_prediction(self):
        print('-------------------------------------------------------------------')
        print('-------------------------------------------------------------------')
        print('Test that mse is zero when y_pred=y_input')
        print('-------------------------------------------------------------------')
        # Make data
        x = np.linspace(0, 10, 100)
        y = 2 + 3*x 

        # Use y as prediction for y
        y_pred = y
        mse = MSE(y, y_pred)
        print(f'Mean squared error when using y as y_pred is {mse}')

        self.assertEqual(mse, 0, 'mse not equal to 0 when using y as y_pred!')

    def test_R2_when_predicting_mean(self):

        # Test that R^2 score is 0 when always predicting the mean
        print('-------------------------------------------------------------------')
        print('-------------------------------------------------------------------')
        print('Test that R2 score is zero when using the mean value as prediction')
        print('-------------------------------------------------------------------')
        # Make data
        x = np.linspace(0, 10, 100)
        y = 2 + 3*x 

        # Use mean value as prediction for y
        y_pred = np.array([np.mean(y) for i in range(len(y))])
        r2_score = R2(y, y_pred)
        print(f'R2 score when using mean value as prediction is {r2_score}')

        self.assertEqual(r2_score, 0, 'R^2 score not zero when always predicting the mean!')

if __name__ == '__main__':
    unittest.main()
