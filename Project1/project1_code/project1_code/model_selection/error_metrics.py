import numpy as np

def R2(y_data, y_model):
    """
    Calculates the R^2 score of the model. An R^2 score of 1 indicates a perfect fit.

    Parameters:
        y_data (array): The real output values.
        y_model (array): The output values predicted by the model. 
  
    Returns:
        R^2 (float): R^2 score of the model
    """

    return 1 - np.sum((y_data - y_model) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2)


def MSE(y_data,y_model):
    """
    Calculates the mean squared error of the model.

    Parameters:
        y_data (array): The real output values.
        y_model (array): The output values predicted by the model. 
  
    Returns:
        MSE (float): mean squared error of the model
    """

    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n