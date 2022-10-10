import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Import self-made modules
from project1_code.linear_model import LinearModel
from project1_code.make_and_prepare_data import FrankeFunction, create_design_matrix
from project1_code.model_selection import R2, MSE, cross_validation

# Import plotting functions
from project1_code.plotting import plot_mse_per_poldegree, plot_R2_per_poldegree

# Choose hyperparameters for model
pol_degree = 5
data_size = 600
test_size = 0.2
error_std = 0.1

# Make data
np.random.seed(5)
x = np.random.uniform(0, 1, data_size)
y = np.random.uniform(0, 1, data_size)

z = FrankeFunction(x, y) + np.random.normal(loc=0, scale=error_std)

# Define empty list in which to store the MSE
mses_test = []
mses_train = []

X = create_design_matrix(x, y, pol_degree)

folds = range(5, 11)

# Test for 5 - 10 folds
for i in folds:
    mse_test, mse_train = cross_validation(X, z, i)
    mses_test.append(mse_test)
    mses_train.append(mse_train)

# Find MSE also from sklearn
mse_test_sk = []

lr_sk = LinearRegression()

for i in folds:
    scores = cross_val_score(lr_sk, X, z, cv=i, scoring='neg_mean_squared_error')
    mse_test_sk.append(-np.mean(scores))

# Write results to file
import os
script_path = os.path.dirname(os.path.realpath(__file__))

with open(script_path + '\\..\\..\\output\\benchmarks\\crossval_sklearn_compare_mse_pol_degree=' + str(pol_degree) + '_datapoints=' + str(data_size) + '.txt', 'w') as f:
    f.write('Number of folds, MSE (homemade), MSE (sklearn) \n')

    for i in range(len(folds)):
         f.write(str(folds[i]) + ', ' + str(round(mses_test[i], 8)) + ', ' + str(round(mse_test_sk[i], 8)) + '\n')