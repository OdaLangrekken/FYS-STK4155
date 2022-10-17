import os
import sys


import numpy as np
from project2_code.regression_methods import OLS, ridge

class LinearModel():

    """

    Attributes
    ----
       coeffs (array): coefficients of the linear model
    
    """

    def __init__(self, regr_type = 'OLS', lamb = 0):
        self.regr_type = regr_type
        self.lamb = lamb
    
    def fit(self, X, y):
        if self.regr_type == 'OLS':
            # Add bias term
            X = self.add_bias(X)
            self.coeffs = OLS(X, y)
        elif self.regr_type == 'ridge':
            X = self.add_bias(X)
            self.coeffs = ridge(X, y, self.lamb)
        
    def predict(self, new_data):
        """
        Uses the linear model to predict the value of new data
        """
        # Add bias term to new data
        new_data = self.add_bias(new_data)
        return new_data @ self.coeffs
         
        
    def add_bias(self, X):
        n = len(X)  # Number of observations in dataset
        # Make column of ones
        X_bias = np.ones((n, 1))
        # Add bias
        X = np.hstack([X_bias, X])
        return X