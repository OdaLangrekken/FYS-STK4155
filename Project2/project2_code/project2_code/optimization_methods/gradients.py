import numpy as np

def gradient_linear(X, y, beta):
    """
    Function that returns the gradient of the cost funtion for linear regression
    """
    n = len(X)
    return (2/n) *X.T @ (X @ beta - y)