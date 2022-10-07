import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Import self-made modules
from project1_code.linear_model import LinearModel
from project1_code.make_and_prepare_data import FrankeFunction, create_design_matrix
from project1_code.model_selection import R2, MSE


# Choose hyperparameters for model
pol_degree = 5
data_size = 200
test_size = 0.2

# Make data
np.random.seed(5)
x = np.random.uniform(0, 1, data_size)
y = np.random.uniform(0, 1, data_size)

z = FrankeFunction(x, y)

# Train and test model for polynomial degree of 5
X = create_design_matrix(x, y, polynomial_degree=pol_degree)
    
# Split data in train and test
X_train, X_test, z_train, z_test = train_test_split(X, z, test_size = test_size)
    
# Train model
lr = LinearModel()
lr.fit(X_train, z_train)

# Find coefficients for homemade model
homemade_coefs = lr.coeffs

# Find coefficients from sklearn
lr_sk = LinearRegression()
lr_sk.fit(X_train, z_train)
sk_intercept = lr_sk.intercept_
sk_coeffs = lr_sk.coef_

coefficient_names = X_train.columns

# Write results to file
import os
script_path = os.path.dirname(os.path.realpath(__file__))

with open(script_path + '\\..\\..\\output\\benchmarks\\osl_sklearn_compare.txt', 'w') as f:
    f.write('Coefficient, Coefficients (OLS_homemade), Coefficients (sklearn) \n')

    f.write('Bias, ' + str(homemade_coefs[0]) + ', ' + str(sk_intercept) + '\n')
    for i in range(1, len(homemade_coefs)):
         f.write(str(coefficient_names[i-1]) + ', ' + str(homemade_coefs[i]) + ', ' + str(sk_coeffs[i-1]) + '\n')