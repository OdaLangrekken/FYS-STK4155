import numpy as np

def gradient_linear(X, y, beta):
    """
    Function that returns the gradient of the squared error cost function
    """
    n = len(X)
    return (2/n) * X.T @ (X @ beta - y)

def gradient_ridge(X, y, beta, lamb):
    """
    Function that returns the gradient of the squared error cost function for Ridge regression
    """
    n = len(X)
    return (2/n) * X.T @ (X @ beta - y) + 2*lamb*beta