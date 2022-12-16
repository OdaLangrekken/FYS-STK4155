import numpy as np
from project3_code import OLS, ridge
from project3_code import gradient_descent

class LinearModel():

    """
    A class to build a Linear Regression Model.

    Attributes
    ----
       regr_type (string): what method to use for regression (Ordinary Least Squares is default)
       fit_type (string): whether to use analytical or numerical method to get the coefficients
       lamb (float): regularization parameter if using Ridge regression
    
    """

    def __init__(self, regr_type = 'OLS', fit_type='analytic', lamb = 0):
        self.regr_type = regr_type
        self.fit_type = fit_type
        self.lamb = lamb
    
    def fit(self, X, y):
        """
        Method that finds the best coefficients beta by fitting the model to the data X and output y

        Parameters
        ----------
            X (dataframe / array): input data
            y (array): target        
        """
        if self.regr_type == 'OLS':
            # Add bias term
            X = self.add_bias(X)

            # Check if coefficients are to be found analytically
            if self.fit_type == 'analytic':
                self.coeffs = OLS(X, y)
            elif self.fit_type == 'GD':
                self.coeffs = gradient_descent(X, y, gradient='linear')
        elif self.regr_type == 'ridge':
            X = self.add_bias(X)
            # Check if coefficients are to be found analytically
            if self.fit_type == 'analytic':
                self.coeffs = ridge(X, y, self.lamb)
            elif self.fit_type == 'GD':
                self.coeffs = gradient_descent(X, y, gradient='ridge')

        
    def predict(self, new_data):
        """
        Uses the linear model to predict the value of new data
        """
        # Add bias term to new data
        new_data = self.add_bias(new_data)
        return new_data @ self.coeffs
         
        
    def add_bias(self, X):
        n = len(X)  # Number of observations in dataset
        # Make column of ones
        X_bias = np.ones((n, 1))
        # Add bias
        X = np.hstack([X_bias, X])
        return X