import pandas as pd

def create_design_matrix(x, y, polynomial_degree=1):
    """
    Creates design matrix for two variables x and y, with polynomial terms up to and including polynomial_degree (default 1)
    """
    
    # Initiate empty matrix
    X = pd.DataFrame()
    for i in range(0, polynomial_degree + 1):
        for j in range(0, polynomial_degree + 1 - i):
            if i == 0 and j == 0:
                continue
            if i == 0:
                col_name = 'y^' + str(j)
            elif j == 0:
                col_name = 'x^' + str(i)
            else:
                col_name = 'x^' + str(i) + '*y^' + str(j)  
            X[col_name] = x**i * y**j
    return X