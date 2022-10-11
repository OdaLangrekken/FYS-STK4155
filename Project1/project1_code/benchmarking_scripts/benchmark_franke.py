import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Import self-made modules
from project1_code.linear_model import LinearModel
from project1_code.make_and_prepare_data import FrankeFunction, create_design_matrix
from project1_code.model_selection import R2, MSE

# Parameters for benchmark model
pol_degree = 5
data_size = 600
test_size = 0.2
error_std = 0.1
num_folds = 5
num_iterations = 100


# Make data
np.random.seed(5)

x = np.random.uniform(0, 1, data_size)
y = np.random.uniform(0, 1, data_size)

z = FrankeFunction(x, y) + np.random.normal(scale = error_std, size = data_size)

# Make design matrix
X = create_design_matrix(x, y, pol_degree)
    
# Split data in train and test
X_train, X_test, z_train, z_test = train_test_split(X, z, test_size = test_size, random_state=1)
    

## OLS    
# Train model
lr = LinearModel()
lr.fit(X_train, z_train)
    
# Made predictions on train and test set
z_pred_test = lr.predict(X_test)
z_pred_train = lr.predict(X_train)

mse_train = MSE(z_train, z_pred_train)
mse_test = MSE(z_test, z_pred_test)

# Write results to file
import os
script_path = os.path.dirname(os.path.realpath(__file__))

with open(script_path + '\\..\\..\\output\\benchmarks\\benchmark_pol_degree=' + str(pol_degree) + '_datapoints=' + str(data_size) + '.txt', 'w') as f:
    f.write('MSE from OLS \n')
    f.write(f'MSE train: {round(mse_train, 8)} \n')
    f.write(f'MSE test: {round(mse_test, 8)} \n')
    f.write(f'With boostrap, iterations = {num_iterations} \n')

    f.write(f'With cross-validation, folds = {num_folds} \n')


