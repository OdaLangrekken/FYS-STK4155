import numpy as np

def cost_linear(X, y, coeffs):
    n = len(y)
    cost = 1/n * np.sum(X @ coeffs - y)**2
    return cost

def cost_ridge(X, y, coeffs, lamb):
    n = len(y)
    cost = 1/n * np.sum(X @ coeffs - y)**2 + lamb * np.sum(coeffs)**2
    return 
    
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