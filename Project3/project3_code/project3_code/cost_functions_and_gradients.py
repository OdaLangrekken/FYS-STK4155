import numpy as np
from project2_code import sigmoid

def cost_linear(X, y, coeffs, lamb=0):
    """
    Cost function for linear regression without regularization. The cost function is the mean squared error
    """
    n = len(y)
    cost = (1/n) * np.sum((X @ coeffs - y)**2) + 1/n * lamb * np.sum(coeffs)**2
    return cost

def cross_entropy(X, y, coeffs, lamb=0):
    """
    Cost function for logisitc regression. The cost function is the cross-entropy (negative log-likelihood)
    """
    n = len(y)
    z = X @ coeffs
    cost = np.sum(y @ z - np.log(1 + np.exp(z))) + lamb * np.sum(coeffs)**2
    return cost
    
def gradient_linear(X, y, beta, lamb=0):
    """
    Function that returns the gradient of the squared error cost function
    """
    n = len(X)
    return (2/n) * X.T @ (X @ beta - y) + (2/n)*lamb*beta

def gradient_logistic(X, y, beta, lamb=0):
    """
    Function that returns the gradient of the cross entropy for logistic regression
    """
    n = len(y)
    # Find vector of fitted probabilites
    p = sigmoid(X @ beta)
    return -X.T @ (y - p) +  lamb*beta
