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

# Define empty list in which to store the MSE and R2 errors
mses = []
mses_train = []
r2s = []
r2s_train = []

# Train and test model for different polynomial derees
for i in range(1, pol_degree+1):
    X = create_design_matrix(x, y, i)
    
    # Split data in train and test
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size = test_size)
    
    # Train model
    lr = LinearModel()
    lr.fit(X_train, z_train)
    
    # Made predictions on train and test set
    z_pred_test = lr.predict(X_test)
    z_pred_train = lr.predict(X_train)
    
    # Calculate errors for test set
    mses.append(MSE(z_test, z_pred_test))
    r2s.append(R2(z_test, z_pred_test))
    
    # Calculate errors for train set
    mses_train.append(MSE(z_train, z_pred_train))
    r2s_train.append(R2(z_train, z_pred_train)) 

# Write results to file
import os
script_path = os.path.dirname(os.path.realpath(__file__))

with open(script_path + '\\..\\..\\output\\benchmarks\\osl_sklearn_compare.txt', 'w') as f:
    f.write('Coefficient, Coefficients (OLS_homemade), Coefficients (sklearn) \n')
    homemade_coefs = lr.coeffs
    
    lr_sk = LinearRegression()
    lr_sk.fit(X_train, z_train)
    sk_intercept = lr_sk.intercept_
    sk_coeffs = lr_sk.coef_

    coefficient_name = X_train.columns
    
    f.write('Bias, ' + str(homemade_coefs[0]) + ', ' + str(sk_intercept) + '\n')
    for i in range(1, len(homemade_coefs)):
         f.write(str(coefficient_name[i-1]) + ', ' + str(homemade_coefs[i]) + ', ' + str(sk_coeffs[i-1]) + '\n')