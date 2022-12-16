import numpy as np
from project3_code import OLS, ridge, sigmoid
from project3_code import gradient_descent, stochastic_gradient_descent

class LogisticRegression:
    
    """
    A class to build a Linear Regression Model.

    Attributes
    ----
       num_classes (int): number of different classification outputs
       regr_type (string): what method to use for regularization (default None)
       solver (string): optimazitation method to use for fitting (default gradient descent)
       learning_rate (float): learing rate to use for optimization method
       epochs (int): number of iterations for gradient descent
       lamb (float): regularization parameter if using regularization
    
    """
        
    def __init__(self, num_classes=2, solver = 'GD', num_batches=1, learning_rate=0.1, epochs = 1000, lamb = 0, random_state=None):
        self.num_classes = num_classes
        self.solver = solver
        self.lamb = lamb
        self.epochs = epochs
        self.num_batches = num_batches
        self.learning_rate = learning_rate
        self.random_state = random_state
                            
    def fit(self, X, y, return_cost=False, X_val=None, y_val=None):
        """
        Method that finds the best coefficients beta by fitting the model to the data X and output y

        Parameters
        ----------
            X (dataframe / array): input data
            y (array): target 
            return_cost (bool): whether to return the cost function as a result of number of iterations
            X_val (dataframe): validtion data
            y_val (array): validation output  
        """
        # Add bias
        X = self.add_bias(X)
        if return_cost:
            X_val = self.add_bias(X_val)
        # Do one fit if we only have two different outputs
        if self.num_classes == 2:
            if self.solver == 'GD':
                self.coeffs = gradient_descent(X, y, loss='logistic', max_iterations=self.epochs, lamb=self.lamb, return_cost=return_cost, X_val=X_val, y_val=y_val)
            elif self.solver == 'SGD':
                self.coeffs = stochastic_gradient_descent(X, y, loss='logistic', alpha=self.learning_rate, num_batches=self.num_batches, epochs=self.epochs, lamb=self.lamb, return_cost=return_cost, X_val=X_val, y_val=y_val)[0]
        # Otherwise fit coefficients for each class
        else:
            self.coeffs = np.ndarray(shape=(X.shape[1], self.num_classes), dtype=float)
            self.cost_train = np.ndarray(shape=(self.epochs, self.num_classes), dtype=float)
            self.cost_test = np.ndarray(shape=(self.epochs, self.num_classes), dtype=float) 
            for c in range(self.num_classes):
                #print(f'Done fitting coefficients for {c}/{self.num_classes} classes')
                y_c = np.array([y_val == c for y_val in y])
                if self.solver == 'GD':
                    self.coeffs[:,c] = gradient_descent(X, y_c, loss='logistic', max_iterations=self.epochs, lamb=self.lamb, return_cost=return_cost, X_val=X_val, y_val=y_val)
                elif self.solver == 'SGD':
                    stochastic_result = stochastic_gradient_descent(X, y_c, loss='logistic', alpha=self.learning_rate, num_batches=self.num_batches, epochs=self.epochs, lamb=self.lamb, return_cost=return_cost, X_val=X_val, y_val=y_val, random_state=self.random_state)
                    self.coeffs[:,c] = stochastic_result[0]
                    if return_cost:
                        self.cost_train[:, c] = stochastic_result[1]
                        self.cost_test[:, c] = stochastic_result[2]

    def predict_proba(self, new_data):
        """
        Uses the logistic model to predict the class probability of new data
        """
        # Add bias term to new data
        new_data = self.add_bias(new_data)
        return sigmoid(new_data @ self.coeffs)

    def predict(self, new_data):
        """
        Uses the logistic model to predict the class of new data
        """
        new_data_probs = self.predict_proba(new_data)
        new_data_class = []
        for prob in new_data_probs:
            new_data_class.append(np.argmax(prob))
        return new_data_class
        
    def add_bias(self, X):
        n = len(X)  # Number of observations in dataset
        # Make column of ones
        X_bias = np.ones((n, 1))
        # Add bias
        X = np.hstack([X_bias, X])
        return X


      
        
