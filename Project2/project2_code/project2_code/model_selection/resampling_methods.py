import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


from project2_code.model_classes import LinearModel
from project2_code.model_selection import MSE

def cross_validation(model, X, z, num_folds=5, scale=False):
    """
    Function that uses cross-validation to compute test MSE
    by splitting the data in num_folds folds and computes test error on 
    ith fold by training on all except the ith fold. This is repeated for all num_folds.
    The test error is computed as the average error over all folds.
    
    Input
    -----
        model (LinearModel class instance): model to be trained
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

    if ~isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    
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

        if scale:
            scaler = StandardScaler()
            scaler.fit(X_train)

            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)

        # Train the model
        model.fit(X_train, z_train)
        
        # Make predictions
        z_train_predict = model.predict(X_train)
        z_test_predict = model.predict(X_test)
        
        # Compute mean squared error for fold
        mse_test.append(MSE(z_test, z_test_predict))
        mse_train.append(MSE(z_train, z_train_predict))
    
    # Compute and return average mean squared error
    mse_test = np.mean(mse_test)
    mse_train = np.mean(mse_train)
    return mse_test, mse_train
            

def make_bootstrap_sample(X, z, sample_size=1):
    """
    Function that generates one bootstrap sample.
    
    Input
    -----
    X (dataframe or matrix): design matrix containing all input data
    z (array): array of outputs
    sample_size (float): percentage of input to use for bootstrap sample
    
    Returns
    -------
    X_sample (dataframe): bootstrap sample of input data
    z_sample (array): output data corresponding to input data in bootstrap sample
    X_test (dataframe): input data not sampled, used as test data
    z_test (dataframe): output data corresponding to input data not sampled
    """
    X = X.reset_index(drop=True)
    # Randomly draw n rows from design matrix X with replacement
    X_sample = X.sample(n=sample_size*len(X), replace=True)
    rows_chosen = X_sample.index
    # Choose same rows from z to get output training data
    z_sample = z[rows_chosen]
    
    # Use rows not sampled as test set
    #X_test = X[~X.index.isin(rows_chosen)]
    #z_test = np.delete(z, rows_chosen)
    
    return X_sample, z_sample

def bootstrap(model, X_train, z_train, X_test, z_test, num_iterations=5, sample_size=1):
    """
    Function that uses bootstrap resampling to compute test and train mean squared error.
    For each iteration a n datapoints (determined by sample_size) are sampled from X and z.
    The model is trained on the sample, and the test error is computed on data not in sample.
    The test and train errors are computed as the average error over all iterations.


    Input
    -----
    X (dataframe or matrix): design matrix containing all input data
    z (array): array of outputs
    num_iterations (int): number of times to do bootstrap 
    sample_size (float): percentage of input to use for bootstrap sample
    
    Returns
    -------
    mse_test (float): average mean squared error on test set
    mse_train (float): average mean squared error on training set
    """

    # Define empty lists to store mean squared errors
    mse_test = []
    mse_train = []
    
    # Do sampling n times
    for i in range(num_iterations):
        X_sample, z_sample = make_bootstrap_sample(X_train, z_train, sample_size=1)
        
        # Train the model
        model.fit(X_sample, z_sample)
        
        # Make predictions
        z_sample_predict = model.predict(X_sample)
        z_test_predict = model.predict(X_test)
        
        # Compute mean squared error for fold
        mse_test.append(MSE(z_test, z_test_predict))
        mse_train.append(MSE(z_sample, z_sample_predict))
    
    # Compute and return average mean squared error
    mse_test = np.mean(mse_test)
    mse_train = np.mean(mse_train)
    return mse_test, mse_train