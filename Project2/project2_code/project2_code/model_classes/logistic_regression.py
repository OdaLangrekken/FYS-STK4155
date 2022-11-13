import numpy as np
from project2_code.regression_methods import OLS, ridge
from project2_code.optimization_methods import gradient_descent

class LogisticRegression:
        
        def __init__(self, num_classes = 2):
            self.num_classes = num_classes
                            
        def fit(self, X, y, regr_type = 'OLS', lamb = 0, alpha = 0.1, max_iters = 100):
            m = X.shape[0]
            n = X.shape[1]

            self.coeffs = gradient_descent(X, y, gradient='linear')

