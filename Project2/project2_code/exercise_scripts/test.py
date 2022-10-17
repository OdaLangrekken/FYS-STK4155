import numpy as np
import matplotlib.pyplot as plt

from project2_code.make_and_prepare_data import simple_polynomial

x = np.random.uniform(0, 10, size = 1000)
x = np.linspace(0, 10, 1000)

# Create polynomial function of x, up to a degree of 5
y = simple_polynomial(x, polynomial_degree = 5)

plt.plot(x, y)
plt.show()
