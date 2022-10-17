import matplotlib.pyplot as plt
import os
current_path = os.getcwd()

def plot_R2_per_poldegree(error_lists, label_lists, pol_degree, save_plot=False, save_title=''):
    ax = plt.subplot()
    for i in range(len(error_lists)):
        plt.plot(range(1, pol_degree+1), error_lists[i], label = label_lists[i]);
    plt.legend();
    plt.xlabel('Polynomial degree');
    plt.ylabel('R$^2$ score');
    ax.set_xticks(range(1, pol_degree+1))

    if save_plot:
        plt.savefig(current_path + '\\project2\\output\\figures\\' + save_title + '.jpg')

def plot_mse_per_poldegree(error_lists, label_lists, pol_degree, save_plot=False, save_title=''):
    ax = plt.subplot()
    for i in range(len(error_lists)):
        plt.plot(range(1, pol_degree+1), error_lists[i], label = label_lists[i]);
    plt.legend();
    plt.xlabel('Polynomial degree');
    plt.ylabel('Mean squared error');
    ax.set_xticks(range(1, pol_degree+1))
    #plt.yscale('log')

    
    if save_plot:
        plt.savefig(current_path + '\\project2\\output\\figures\\' + save_title + '.jpg')