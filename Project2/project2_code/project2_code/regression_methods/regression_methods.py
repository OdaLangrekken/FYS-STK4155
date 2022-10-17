import numpy as np 
from project1_code.make_and_prepare_data import FrankeFunction

def OLS(X, y):
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

def ridge(X, y, lamb):
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