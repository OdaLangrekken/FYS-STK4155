import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Import self-made modules
from project1_code.linear_model import LinearModel
from project1_code.make_and_prepare_data import FrankeFunction, create_design_matrix
from project1_code.model_selection import R2, MSE, bootstrap

# Import plotting functions
from project1_code.plotting import plot_mse_per_poldegree, plot_R2_per_poldegree

# Parameters for benchmark model
pol_degree = 5
data_size = 600
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

mse_test, mse_train = bootstrap(X, z, 5)

print(mse_train)
print(mse_test)