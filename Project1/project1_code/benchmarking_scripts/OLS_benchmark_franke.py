import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Import self-made modules
from project1_code.linear_model import LinearModel
from project1_code.make_and_prepare_data import FrankeFunction, create_design_matrix
from project1_code.model_selection import R2, MSE

# Import plotting functions
from project1_code.plotting import plot_mse_per_poldegree, plot_R2_per_poldegree

# Parameters for benchmark model
pol_degree = 5
data_size = 300
test_size = 0.2
error_std = 0.1
save_fig = False

# Make data
np.random.seed(5)

x = np.random.uniform(0, 1, data_size)
y = np.random.uniform(0, 1, data_size)

z = FrankeFunction(x, y) + np.random.normal(scale = error_std, size = data_size)

# Make design matrix
X = create_design_matrix(x, y, pol_degree)
    
# Split data in train and test
X_train, X_test, z_train, z_test = train_test_split(X, z, test_size = test_size)
    
# Train model
lr = LinearModel()
lr.fit(X_train, z_train)
    
# Made predictions on train and test set
z_pred_test = lr.predict(X_test)
z_pred_train = lr.predict(X_train)

mse_train = MSE(z_train, z_pred_train)
mse_test = MSE(z_test, z_pred_test)

r2_train = R2(z_train, z_pred_train)
r2_test = R2(z_test, z_pred_test)

print(mse_train)
print(mse_test)
print(r2_train)
print(r2_test)
