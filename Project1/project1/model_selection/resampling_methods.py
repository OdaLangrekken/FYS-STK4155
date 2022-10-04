import os
import sys

current_path = os.getcwd()
sys.path.append(current_path + '\Project1\project1')

import pandas as pd
import numpy as np
from linear_model import LinearModel
from model_selection import MSE

def cross_validation(X, z, num_folds=5):
    """
    Function that uses cross-validation to compute test MSE
    by splitting the data in num_folds folds and computes test error on 
    ith fold by training on all except the ith fold. This is repeated for all num_folds.
    The test error is computed as the average error over all folds.
    
    Input
    -----
        X (dataframe): the design matrix containing all input data
        z (array): an array of outputs
        num_folds (int): number of folds
        
    Returns
    -----
       mse_test (float): average mean squared error on test set
       mse_train (float): average mean squared error on training set
    """
    # Define empty lists to store mean squared errors
    mse_test = []
    mse_train = []
    
    # Find size of each fold
    fold_size = int(len(X) / num_folds)
    
    # Loop thorough all folds
    for i in range(num_folds):
        # Select only the ith fold of the data for the test set
        X_test = X.iloc[i*fold_size:(i+1)*fold_size]
        z_test = z[i*fold_size:(i+1)*fold_size]
                        
        # Make sure all data is included if len_data/num_folds is uneven
        if i == num_folds-1:
            X_test = X.iloc[i*fold_size:]
            z_test = z[i*fold_size:]
        
        # Select the rest of the data for the training set
        X_train = pd.concat([X.iloc[:i*fold_size], X.iloc[(i+1)*fold_size:]])
        z_train = np.concatenate([z[:i*fold_size], z[(i+1)*fold_size:]])

        # Train the model
        lm = LinearModel()
        lm.fit(X_train, z_train)
        
        # Make predictions
        z_train_predict = lm.predict(X_train)
        z_test_predict = lm.predict(X_test)
        
        # Compute mean squared error for fold
        mse_test.append(MSE(z_test, z_test_predict))
        mse_train.append(MSE(z_train, z_train_predict))
    
    # Compute and return average mean squared error
    mse_test = np.mean(mse_test)
    mse_train = np.mean(mse_train)
    return mse_test, mse_train
            