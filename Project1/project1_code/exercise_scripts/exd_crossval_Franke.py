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

# Choose parameters for model
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

# Test for 5 - 10 folds
for i in range(5, 11):
    model = LinearModel()
    mse_test, mse_train = cross_validation(model, X, z, i)
    mses_test.append(mse_test)
    mses_train.append(mse_train)

plt.plot(range(5, 11), mses_test, label='Test')
plt.xlabel('Number of folds')
plt.ylabel('Mean squared error')
plt.show()

