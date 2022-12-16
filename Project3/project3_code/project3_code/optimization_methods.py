import numpy as np
from sklearn.utils import shuffle
from project3_code import gradient_linear, gradient_logistic
from project3_code import cost_linear, cross_entropy, accuracy, sigmoid

def gradient_descent(X, y, alpha=0.1, max_iterations = 100, loss='squared_error', lamb=0, momentum_param=0, return_cost=False, X_val=None, y_val=None):
    """
    Function that uses gradient descent to find the coefficients that minimze the loss function.

    Input
    ------------
    X (dataframe): input data
    y (array): output
    alpha (float): learning rate
    max_iterations (int): number of iterations for gradient descent
    loss (string): loss function to minimize. Default is squared_error
    lamb (float): regularization parameter for Ridge regression
    momentum_param (float): momentum parameter, default 0 (no momentum)
    return_cost (bool): whether to return the cost function as a result of number of iterations
    X_val (dataframe): validtion data
    y_val (array): validation output

    Returns
    -----------
    array: optimized coefficients
    """
    coeff_num = X.shape[1]   
    # Initialize random coefficients
    coeffs = np.random.randn(coeff_num)

    # Initialize empty list for cost
    cost_train = []
    cost_val = []

    last_update = 0
    
    # Change coefficients in direction of biggest gradient for max_iterations iterations
    for iteration in range(max_iterations):
        # Find gradient
        if loss == 'squared_error':
            gradient = gradient_linear(X, y, coeffs, lamb)
            if return_cost:
                cost_train.append(cost_linear(X, y, coeffs, lamb))
                cost_val.append(cost_linear(X_val, y_val, coeffs, lamb))
        elif loss == 'logistic':
            gradient = gradient_logistic(X, y, coeffs, lamb)
            if return_cost:
                cost_train.append(cross_entropy(X, y, coeffs, lamb))
                cost_val.append(cross_entropy(X_val, y_val, coeffs, lamb))
        # Update coefficients
        update = momentum_param*last_update + alpha*gradient
        coeffs = coeffs - update
        last_update = update

    if return_cost:
        return coeffs, cost_train, cost_val
    return coeffs


def stochastic_gradient_descent(X, y, alpha, num_batches, epochs, random_state=None, loss='squared_error', lamb=0, momentum_param=0, return_cost=False, X_val=None, y_val=None):
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
    lamb (float): regularization parameter for Ridge regression
    momentum_param (float): momentum parameter, default 0 (no momentum)
    return_cost_val (bool): whether to return the cost function as a result of number of iterations 
    X_val (dataframe): validtion data
    y_val (array): validation output

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
    cost_train = []
    cost_val = []

    last_update = 0

    # Shuffle data
    X_shuffle, y_shuffle = shuffle(X, y, random_state=random_state)
    
    for epoch in range(epochs):
        
        # Choose random batch
        np.random.seed(random_state)
        batch_chosen = np.random.randint(0, num_batches)
        Xi = X_shuffle[batch_chosen:batch_chosen+batch_size]
        yi = y_shuffle[batch_chosen:batch_chosen+batch_size]
        # Find gradient for given cost function
        if loss == 'squared_error':
            gradient = gradient_linear(Xi, yi, coeffs, lamb)
            if return_cost:
                cost_train.append(cost_linear(X, y, coeffs, lamb))
                cost_val.append(cost_linear(X_val, y_val, coeffs, lamb))
        elif loss == 'logistic':
            gradient = gradient_logistic(Xi, yi, coeffs, lamb)
            if return_cost:
                y_model = sigmoid
                cost_train.append(cross_entropy(X, y, coeffs, lamb))
                cost_val.append(cross_entropy(X_val, y_val, coeffs, lamb))
        # Update coefficients
        update = momentum_param*last_update + alpha*gradient
        coeffs = coeffs - update
        last_update = update
            
    return coeffs, cost_train, cost_val