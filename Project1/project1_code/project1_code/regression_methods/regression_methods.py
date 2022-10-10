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
    # Center input
    #X = X - np.mean(X, axis=0)
    # Solve for the coefficents  
    coeffs = np.linalg.inv(X.T @ X + lamb*np.identity(p)) @ X.T @ y
    # Find bias
    ##y_pred = X @ coeffs
    #beta0 = np.sum(y_pred)/len(y_pred)
    #print(beta0)
    #coeffs = np.append([beta0], coeffs)
    return coeffs