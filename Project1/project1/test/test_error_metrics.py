import os
import sys

current_path = os.getcwd()
sys.path.append(current_path + '\Project1\project1')

from linear_model import LinearModel
from model_selection import MSE, R2
import unittest
import numpy as np
import pandas as pd

class TestLinearModel1(unittest.TestCase):

    def test_linear_model_simple(self):

        # Test that R^2 score is 0 when always predicting the mean
        print('Test that R2 score is zero when using the mean value as prediction')
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
