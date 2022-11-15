import numpy as np
from sklearn.utils import shuffle
from project2_code import gradient_linear, gradient_ridge, gradient_logistic
from project2_code import cost_linear, cost_ridge, cross_entropy

def gradient_descent(X, y, alpha=0.1, max_iterations = 100, loss='squared_error', return_cost=False, lamb=0.1, momentum_param=0):
    """
    Function that uses gradient descent to find the coefficients that minimze the loss function.

    Input
    ------------
    X (dataframe): input data
    y (array): output
    alpha (float): learning rate
    max_iterations (int): number of iterations for gradient descent
    loss (string): loss function to minimize. Default is squared_error
    return_cost (bool): whether to return the cost function as a result of number of iterations
    lamb (float): regularization parameter for Ridge regression
    momentum_param (float): momentum parameter, default 0 (no momentum)

    Returns
    -----------
    array: optimized coefficients
    """
    coeff_num = X.shape[1]   
    # Initialize random coefficients
    coeffs = np.random.randn(coeff_num)

    # Initialize empty list for cost
    cost = []

    last_update = 0
    
    # Change coefficients in direction of biggest gradient for max_iterations iterations
    for iteration in range(max_iterations):
        # Find gradient
        if loss == 'squared_error':
            gradient = gradient_linear(X, y, coeffs)
            if return_cost:
                cost.append(cost_linear(X, y, coeffs))
        elif loss == 'squared_error_ridge':
            gradient = gradient_ridge(X, y, coeffs, lamb)
            if return_cost:
                cost.append(cost_ridge(X, y, coeffs, lamb))
        elif loss == 'logistic':
            gradient = gradient_logistic(X, y, coeffs)
            if return_cost:
                cost.append(cross_entropy(X, y, coeffs))
        # Update coefficients
        update = momentum_param*last_update + alpha*gradient
        coeffs = coeffs - update
        last_update = update

    if return_cost:
        return coeffs, cost
    return coeffs


def stochastic_gradient_descent(X, y, alpha, num_batches, epochs, random_state=None, loss='squared_error', return_cost=False, lamb=0.1, momentum_param=0):
    """
    Function that uses stochastic gradient descent to find the coefficients that minimize the cost function.

    Input
    ------------
    X (dataframe): input data
    y (array): output
    alpha (float): learning rate
    num_batches (int): the number of mini batches
    epochs (int): number of times to run gradient descent on all minibacthes
    random_state (int): random_state to use for shuffle. Set to int for reproducible results
    loss (string): loss function to minimize. Default is squared_error
    return_cost_val (bool): whether to return the cost function as a result of number of iterations 
    lamb (float): regularization parameter for Ridge regression
    momentum_param (float): momentum parameter, default 0 (no momentum)

    Returns
    -----------
    array: optimized coefficients
    """
    n = len(X)  # number of rows
    batch_size = int(n/num_batches)
      
    # Initialize random coefficients
    coeff_num = X.shape[1] 
    coeffs = np.random.randn(coeff_num)

    # Initialize empty list for cost
    cost = []

    last_update = 0
    
    for epoch in range(epochs):
        # Shuffle data
        X_shuffle, y_shuffle = shuffle(X, y, random_state=random_state)
        # Choose random batch
        batch_chosen = np.random.randint(0, num_batches)
        Xi = X[batch_chosen:batch_chosen+batch_size]
        yi = y[batch_chosen:batch_chosen+batch_size]
        # Find gradient for given cost function
        if loss == 'squared_error':
            gradient = gradient_linear(X, y, coeffs)
            if return_cost:
                cost.append(cost_linear(X, y, coeffs))
        elif loss == 'squared_error_ridge':
            gradient = gradient_ridge(X, y, coeffs, lamb)
            if return_cost:
                cost.append(cost_ridge(X, y, coeffs, lamb))
        elif loss == 'logistic':
            gradient = gradient_logistic(X, y, coeffs)
            if return_cost:
                cost.append(cross_entropy(X, y, coeffs))
        # Update coefficients
        update = momentum_param*last_update + alpha*gradient
        coeffs = coeffs - update
        last_update = update
            
    if return_cost:
        return coeffs, cost
    return coeffs