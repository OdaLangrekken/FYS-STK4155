import matplotlib.pyplot as plt
import os
current_path = os.getcwd()

def plot_bias_variance(biases, variances, mses, pol_degree, save_plot=False, save_title=''):
    ax = plt.subplot()
    
    plt.plot(range(1, pol_degree+1), biases, label='Bias^2')
    plt.plot(range(1, pol_degree+1), variances, label='Variance')
    plt.plot(range(1, pol_degree+1), mses, label='MSE')
    plt.legend()
    plt.xlabel('Polynomial degree');
    plt.ylabel('Error');
    ax.set_xticks(range(1, pol_degree+1))

    if save_plot:
        plt.savefig(current_path + '\\Project1\\output\\figures\\' + save_title + '.jpg')