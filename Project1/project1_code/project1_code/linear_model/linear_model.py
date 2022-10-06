import os
import sys


import numpy as np
from project1_code.regression_methods import OLS

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
        # Add bias term
        X = self.add_bias(X)
        if self.regr_type == 'OLS':
            self.coeffs = OLS(X, y)
        elif self.regr_type == 'ridge':
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
    
    def OLS(self, X, y):
        """
        Function that finds the coefficients that minimize the residual sum of squares.
    
        Parameters:
            x (array of shape (n, m)): array containing the m different features
            y (array of shape (n, 1)): array containing the output
       
        Output:
            coeffs (array of shape (m+1,1)): coefficients that minimize the residual sum of squares
        """
        # Solve for the coefficents  
        coeffs = np.linalg.inv(X.T @ X) @ X.T @ y
        return coeffs

    def ridge(self, X, y, lamb):
        """
        Function that finds the coefficients that minimize the residual sum of squares using ridge regression.
    
        Parameters:
            x (array of shape (n, m)): array containing the m different features
            y (array of shape (n, 1)): array containing the output
       
        Output:
            coeffs (array of shape (m+1,1)): coefficients that minimize the residual sum of squares using ridge regression
        """
        # Find number of features
        p = X.shape[1]
        # Solve for the coefficents  
        coeffs = np.linalg.inv(X.T @ X + lamb*np.identity(p)) @ X.T @ y
        return coeffs