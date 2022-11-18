import numpy as np
from project2_code import OLS, ridge
from project2_code import gradient_descent, stochastic_gradient_descent

class LogisticRegression:
    
    """
    A class to build a Linear Regression Model.

    Attributes
    ----
       regr_type (string): what method to use for regularization (default None)
       solver (string): optimazitation method to use for fitting (default gradient descent)
       learning_rate (float): learing rate to use for optimization method
       max_iterations (int): number of iterations for gradient descent
       lamb (float): regularization parameter if using regularization
    
    """
        
    def __init__(self, solver = 'GD', num_batches=1, learning_rate=0.1, epochs = 1000, lamb = 0):
        self.solver = solver
        self.lamb = lamb
        self.epochs = epochs
        self.num_batches = num_batches
        self.learning_rate = learning_rate
                            
    def fit(self, X, y):
        """
        Method that finds the best coefficients beta by fitting the model to the data X and output y

        Parameters
        ----------
            X (dataframe / array): input data
            y (array): target        
        """
        # Add bias
        X = self.add_bias(X)
        if self.solver == 'GD':
            self.coeffs = gradient_descent(X, y, loss='logistic', max_iterations=self.epochs, lamb=self.lamb)
        elif self.solver == 'SGD':
            self.coeffs = stochastic_gradient_descent(X, y, loss='logistic', alpha=self.learning_rate, num_batches=self.num_batches, epochs=self.epochs, lamb=self.lamb)

    def predict_proba(self, new_data):
        """
        Uses the logistic model to predict the class probability of new data
        """
        # Add bias term to new data
        new_data = self.add_bias(new_data)
        return new_data @ self.coeffs

    def predict(self, new_data):
        """
        Uses the logistic model to predict the class of new data
        """
        # Add bias term to new data
        new_data = self.add_bias(new_data)
        new_data_probs = new_data @ self.coeffs
        prob_converter = lambda prob: 0 if prob < 0.5 else 1
        new_data_class = [prob_converter(prob) for prob in new_data_probs]
        return new_data_class
        
    def add_bias(self, X):
        n = len(X)  # Number of observations in dataset
        # Make column of ones
        X_bias = np.ones((n, 1))
        # Add bias
        X = np.hstack([X_bias, X])
        return X


      
        
