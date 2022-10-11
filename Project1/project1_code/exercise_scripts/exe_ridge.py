import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Import self-made modules
from project1_code.linear_model import LinearModel
from project1_code.make_and_prepare_data import FrankeFunction, create_design_matrix
from project1_code.model_selection import R2, MSE, bootstrap, bias_variance_decomp, cross_validation

# Import plotting functions
from project1_code.plotting import plot_mse_per_poldegree, plot_R2_per_poldegree, plot_bias_variance

# Choose parameters for model, should read these from terminal or scripts
pol_degree = 10
data_size = 600
test_size = 0.2
error_std = 0.1
num_iterations = 100
num_folds = 10
lamb = 0
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
mses_bias_vars = []

mses_test_cv = []
mses_train_cv = []

# Train and test model for different polynomial derees
for i in range(1, pol_degree+1):
    X = create_design_matrix(x, y, i)
    
    # Split data in train and test
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size = test_size, random_state=1)

    # Scale the data

    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    lr = LinearModel(regr_type='ridge', lamb=lamb)
    lr.fit(X_train, z_train)
    
    # Made predictions on train and test set
    z_pred_test = lr.predict(X_test)
    z_pred_train = lr.predict(X_train)
    
    # Calculate errors for test set
    mses_test.append(MSE(z_test, z_pred_test))
    # Calculate errors for train set
    mses_train.append(MSE(z_train, z_pred_train))

    # Calculate MSE using bootstrap
    mse_bs_test, mse_bs_train = bootstrap(lr, X_train, z_train, X_test, z_test, num_iterations, sample_size=1)
    mses_bs_train.append(mse_bs_train)
    mses_bs_test.append(mse_bs_test)

    # Find MSE using cross val
    mse_test_cv, mse_train_cv = cross_validation(lr, X, z, 5, scale=True)
    mses_test_cv.append(mse_test_cv)
    mses_train_cv.append(mse_train_cv)

    # Decompose bias and variance using bootstrap
    bias, variance, mse_bias_var =  bias_variance_decomp(lr, X_train, z_train, X_test, z_test, boostrap_iterations=num_iterations)
    biases.append(bias)
    variances.append(variance)
    mses_bias_vars.append(mse_bias_var)

# Plot MSE
plot_mse_per_poldegree([mses_train, mses_test, mses_bs_train, mses_bs_test], ['Ridge train', 'Ridge test', 'Bootstrap train', 'Bootstrap test'], pol_degree=pol_degree, save_plot = save_fig, save_title = 'bootstrap_ridge_n=' + str(data_size) + ', lambda=' + str(lamb) + ', iterations=' + str(num_iterations))
plt.show()
plt.clf()

plot_mse_per_poldegree([mses_train_cv, mses_test_cv, mses_bs_train, mses_bs_test], ['CV train', 'CV test', 'Bootstrap train', 'Bootstrap test'], pol_degree=pol_degree, save_plot = save_fig, save_title = 'bootstrap_crossval_ridge_n=' + str(data_size) + ', lambda=' + str(lamb) + ', iterations=' + str(num_iterations))
plt.show()
plt.clf()

plot_bias_variance(biases, variances, mses_bias_vars, pol_degree, save_plot=save_fig, save_title='bootstrap_biasVar_ridge_n=' + str(data_size) + ', lambda=' + str(lamb) + ', iterations=' + str(num_iterations))
plt.show()


