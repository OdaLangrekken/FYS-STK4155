import numpy as np
from project2_code.optimization_methods import gradient_linear

def gradient_descent(X, y, alpha=0.1, max_iterations = 1000):
    """
    
    Input
    ------------
    X (dataframe): input data
    y (array): output
    alpha (float): learning rate
    max_iterations (int): number of iterations for gradient descent
    """
    coeff_num = X.shape[1]   
    # Initialize random coefficients
    coeffs = np.random.randn(coeff_num,1)
    
    # Change coefficients in direction of biggest gradient for max_iterations iterations
    for iteration in range(max_iterations):
        # Find gradient
        gradient = gradient_linear(X, y, coeffs)
        # Update coefficients
        coeffs -= alpha*gradient
    return coeffs