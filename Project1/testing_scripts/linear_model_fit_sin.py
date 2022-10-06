import os
import sys

current_path = os.getcwd()
print(current_path)
sys.path.append(current_path + '\Project1\project1')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import self-made modules
from linear_model import LinearModel
from make_and_prepare_data import create_design_matrix_1d
from model_selection import R2, MSE

# Make sine curve with a guassian noise with sigma^2=0.3
np.random.seed(5)
x = np.linspace(0, 2*np.pi, 40)
y = np.sin(x) + np.random.normal(scale = 0.3, size = 40)

plt.scatter(x, y, color='black', label = 'Datapoints')
plt.plot(x, np.sin(x),'--',label='True sin(x)')

# Train model for different polynomial degrees
for i in [1, 3, 13]:
    X = create_design_matrix_1d(x, i)
    
    # Train model
    lr = LinearModel()
    lr.fit(X, y)
    
    # Made predictions 
    y_pred = lr.predict(X)
    
    plt.plot(x, y_pred, label='Pol degree = ' + str(i))
plt.xlabel('x')
plt.legend()
plt.savefig(current_path + '\\Project1\\figures\\sinfit1.jpg')
plt.show()