import matplotlib.pyplot as plt
import os
current_path = os.getcwd()

def plot_R2_per_poldegree(R2_train, R2_test, pol_degree, save_plot=False, save_title=''):
    ax = plt.subplot()
    plt.plot(range(1, pol_degree+1), R2_test, label = 'Test');
    plt.plot(range(1, pol_degree+1), R2_train, label = 'Train');
    plt.legend();
    plt.xlabel('Polynomial degree');
    plt.ylabel('R$^2$ score');
    #plt.title('R$^2$ score as a function of polynomial degree');
    ax.set_xticks(range(1, pol_degree+1))

    if save_plot:
        plt.savefig(current_path + '\\..\\figures\\' + save_title + '.jpg')

def plot_mse_per_poldegree(mse_train, mse_test, pol_degree, save_plot=False, save_title=''):
    ax = plt.subplot()
    plt.plot(range(1, pol_degree+1), mse_test, label = 'Test');
    plt.plot(range(1, pol_degree+1), mse_train, label = 'Train');
    plt.legend();
    plt.xlabel('Polynomial degree');
    plt.ylabel('Mean squared error');
    #plt.title('Mean squared error as a function of polynomial degree');
    ax.set_xticks(range(1, pol_degree+1))
    #plt.yscale('log')

    
    if save_plot:
        plt.savefig(current_path + '\\..\\figures\\' + save_title + '.jpg')