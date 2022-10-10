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

# Choose parameters for model, should read these from terminal or scripts
pol_degree = 10
data_size = 300
test_size = 0.2
error_std = 0.1
num_iterations = 100
save_fig = True

# Make data
np.random.seed(5)

x = np.random.uniform(0, 1, data_size)

y = np.random.uniform(0, 1, data_size)
z = FrankeFunction(x, y) + np.random.normal(scale = error_std, size = data_size)

# Define empty list in which to store the MSE and R2 errors
mses_train = []
mses_test = []
mses_bs_train = []
mses_bs_test = []

# Define lists in which to store variance and squared bias
variances = []
biases = []


# Train and test model for different polynomial derees
for i in range(1, pol_degree+1):
    X = create_design_matrix(x, y, i)
    
    # Split data in train and test
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size = test_size, random_state=1)
    
    # Train model
    lr = LinearModel()
    lr.fit(X_train, z_train)
    
    # Made predictions on train and test set
    z_pred_test = lr.predict(X_test)
    z_pred_train = lr.predict(X_train)
    
    # Calculate errors for test set
    mses_test.append(MSE(z_test, z_pred_test))
    # Calculate errors for train set
    mses_train.append(MSE(z_train, z_pred_train))

    # Calculate MSE using bootstrap
    mse_bs_test, mse_bs_train = bootstrap(X, z, num_iterations, sample_size=1)
    mses_bs_train.append(mse_bs_train)
    mses_bs_test.append(mse_bs_test)



# Plot MSE
plot_mse_per_poldegree([mses_train, mses_test, mses_bs_train, mses_bs_test], ['OLS train', 'OLS test', 'Bootstrap train', 'Bootstrap test'], pol_degree=pol_degree, save_plot = save_fig, save_title = 'bootstrap_OLS_n=' + str(data_size) + ', sigma=' + str(error_std) + ', iterations=' + str(num_iterations))


plt.show()