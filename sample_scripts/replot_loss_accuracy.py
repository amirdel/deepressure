import numpy as np
import matplotlib.pyplot as plt
import os as os

def plot_both(save_path, train, validation, y_label):
    fig, ax = plt.subplots(1, 1)
    fig.hold(True)
    ax.plot(train, label='train')
    ax.plot(validation, label='validation')
    ax.legend(loc='best', fontsize=15)
    ax.set_xlabel('epoch number')
    ax.set_ylabel(y_label)
    fig.savefig(save_path, format='png')
    plt.close(fig)

if __name__ == '__main__':
    # plt.rcParams.update({'font.size': 20})
    # plt.rc('xtick', labelsize=16)
    # plt.rc('ytick', labelsize=16)
    plt.rcParams.update({'figure.autolayout': True})
    result_folder = '/home/amirhossein/projects/deeppressure/results/inception3_dout_08'
    save_folder = '/home/amirhossein/projects/deeppressure/results/inception3_dout_08/partial_plots'
    last_idx = 100
    first_idx = 1
    ## error plot
    error_file = np.load(os.path.join(result_folder, 'error.npz'))
    train_error, validation_error = error_file['train'], error_file['validation']
    plot_path = os.path.join(save_folder, 'accuracy_new.png')
    plot_both(plot_path, 1-np.array(train_error[:last_idx]), 1-np.array(validation_error[:last_idx]),
              'accuracy')
    ## divergence plot
    div_file = np.load(os.path.join(result_folder, 'divergence.npz'))
    train_div_fro, val_div_fro = div_file['train_fro'], div_file['validation_fro']
    train_div_max, val_div_max = div_file['train_max'], div_file['validation_max']
    plot_path = os.path.join(save_folder, 'div_fro_new.png')
    plot_both(plot_path, train_div_fro[:last_idx], val_div_fro[:last_idx], r'$||Div(p)||_{fro}$')
    plot_path = os.path.join(save_folder, 'div_max_new.png')
    plot_both(plot_path, train_div_max[:last_idx], val_div_max[:last_idx], '||Div(p)||')

    ## loss plot
    idx1 = 1*192
    idx2 = 98*192
    loss_file = np.load(os.path.join(result_folder, 'loss.npz'))
    loss, iter_number = loss_file['loss'], loss_file['iter_number']
    fig, ax = plt.subplots(1,1)
    ax.plot(iter_number[idx1: idx2], loss[idx1:idx2])
    fig.savefig(os.path.join(save_folder, 'loss_new.png'))
    ax.set_xlabel('iteration number')
    ax.set_ylabel('loss')
    plt.close(fig)
