import numpy as np

def FrankeFunction(x,y):
    """
    The Franke function, a function of two variables x and y

    Input
    ------
    x (array): input data for variable x
    y (array): input data for variable y

    Returns 
    -------
    array: ouput of the Franke function for the input data
    """
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

def simple_polynomial(x, polynomial_degree, set_seed = True):
    """
    Function that generates a polynomial function, y(x) = a0 + a1*x + a2*x**2 + ... + an*x**n,
    where n is determined by the parameter polynomial_degree

    Input
    -----------
    x (array): input data
    polynomial_degree (int): highest polynomial degree to include
    set_seed (bool): False if no random seed for numpy, True otherwise (default)

    Returns
    -----------
    array: value of polynomial function up to polynomial_degree in the input x
    """
    # Generate random coefficients between -10 and 10
    np.random.seed(5)
    coeffs = np.random.uniform(-10, 10, size = polynomial_degree+1)
    x_polynomial = [x**i for i in range(polynomial_degree+1)]
    y = coeffs @ x_polynomial
    return y

def sigmoid(z):
    """
    The sigmoid function
    """
    # Prevent overflow.
    z = np.clip( z, -500, 500 )
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return np.where(z > 0, 1, 0)

def relu_leaky(z):
    return np.where(z > 0, z, 0.01*z)

def relu_leaky_derivative(z):
    return np.where(z > 0, 1, 0.01)