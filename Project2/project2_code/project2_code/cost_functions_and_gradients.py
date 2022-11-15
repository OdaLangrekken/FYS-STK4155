import numpy as np
from project2_code import sigmoid

def cost_linear(X, y, coeffs):
    """
    Cost function for linear regression without regularization. The cost function is the mean squared error
    """
    n = len(y)
    cost = 1/n * np.sum(X @ coeffs - y)**2
    return cost

def cost_ridge(X, y, coeffs, lamb):
    """
    Cost function for ridge regression.
    """
    n = len(y)
    cost = 1/n * np.sum(X @ coeffs - y)**2 + lamb * np.sum(coeffs)**2
    return 

def cross_entropy(X, y, coeffs):
    """
    Cost function for logisitc regression. The cost function is the cross-entropy (negative log-likelihood)
    """
    z = X @ coeffs
    cost = np.sum(y @ z - np.log(1 + np.exp(z)))
    
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

def gradient_logistic(X, y, coeffs):
    """
    Function that returns the gradient of the cross entropy for logistic regression
    """
    # Find vector of fitted probabilites
    p = sigmoid(X @ coeffs)
    return -X.T @ (y - p)
