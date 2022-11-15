import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Import self-made modules
from project2_code import *

np.random.seed(5)
n = 1000
x = np.random.rand(n)

# Create polynomial function of x, up to a degree of 2
y = simple_polynomial(x, polynomial_degree = 2, )

# Create desgin matrix
X = create_design_matrix_1d(x, 2)
X.insert(0, 'bias', 1)

# Split in train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=1)

# Choose learning rates to plot for gradient descent
learning_rates = [0.1, 0.3, 0.5]
# Choose max iterations for gradient descent
max_iterations = 2000

for learning_rate in learning_rates:
    coeff, cost = gradient_descent(X_train, y_train, alpha=learning_rate, max_iterations=max_iterations, return_cost=True)
    plt.plot(range(max_iterations), cost, label=r'$\alpha$='+str(learning_rate))

plt.yscale('log')
plt.legend()
plt.xlabel('Number of iterations')
plt.ylabel('Mean squared error')
plt.savefig('..\\output\\figures\\learning_rates_GD.jpg')

# Compare convergence when adding momentum
learning_rate=0.5
error_margin = 10E-6
interations_needed = []
momentum_parameters = [0, 0.1, 0.3]
for momentum_param in momentum_parameters:
    coeff, cost = gradient_descent(X_train, y_train, alpha=learning_rate, max_iterations=2000, return_cost=True, momentum_param=momentum_param)
    num_iterations = sum(np.array(cost) > error_margin)
    interations_needed.append(num_iterations)

with open('..\\output\\convergence_compare_GD.txt', 'w') as f:
    f.write(f'Comparing convergence GD with momentum. Learning rate: {learning_rate}. Convergence when error less than: {error_margin}  \n')
    for i in range(len(momentum_parameters)):
        f.write(f'Momentum parameter: {momentum_parameters[i]}. Convergence after {interations_needed[i]} iterations \n')
