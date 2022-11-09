import numpy as np

def cost_linear(X, y, coeffs):
    n = len(y)
    cost = 1/n * np.sum(X @ coeffs - y)**2
    return cost

def cost_ridge(X, y, coeffs, lamb):
    n = len(y)
    cost = 1/n * np.sum(X @ coeffs - y)**2 + lamb * np.sum(coeffs)**2
    return cost