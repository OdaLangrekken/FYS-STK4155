import numpy as np
from project3_code import make_bootstrap_sample, MSE

def bias_variance_decomp(model, X_train, z_train, X_test, z_test, boostrap_iterations=5):

    z_preds = np.empty((z_test.shape[0], boostrap_iterations))

    for i in range(boostrap_iterations):
        X_sample, z_sample = make_bootstrap_sample(X_train, z_train, sample_size=1)

        #Train model
        model.fit(X_sample, z_sample)

        z_pred = model.predict(X_test)
        z_preds[:, i] = z_pred

    error = np.mean( np.mean((z_test - z_preds.T)**2, axis=1, keepdims=True) )
    bias = np.mean( (z_test - np.mean(z_preds, axis=1))**2 )
    variance = np.mean( np.var(z_preds, axis=1, keepdims=True) )

    return bias, variance, error
