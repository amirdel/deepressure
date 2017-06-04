import pickle as pickle
import numpy as np
from deepres.simulator.operators import face_velocity_operator_nonperiodic, get_cell_velocity
import os as os
from os.path import dirname
from copy import copy
import matplotlib.pyplot as plt

def velocity_from_pressure(perm, pressure_tensor, grid, n_max = 10):
    u_list, v_list = [], []
    n_samples = pressure_tensor.shape[0]
    n_cells = grid.nr_p
    # nx = grid.m
    n_samples = min(n_max, n_samples)
    for i in range(n_samples):
        print(i)
        grid.set_transmissibility(perm[i].reshape((n_cells)))
        pressure_vec = pressure_tensor[i,:,:,0].reshape((n_cells))
        U_face_operator = face_velocity_operator_nonperiodic(grid)
        u_face = U_face_operator.dot(pressure_vec)
        u_cell, v_cell = get_cell_velocity(grid, u_face)
        u_list.append(list(u_cell))
        v_list.append(list(v_cell))
    return np.ndarray.flatten(np.array(u_list)), np.ndarray.flatten(np.array(v_list))

def get_cdf(input1):
    # sum of array should be one
    input_array = copy(input1)
    input_array = input_array / float(np.sum(input_array))
    cdf = [input_array[0]]
    for i in range(1, len(input_array)):
        cdf.append(cdf[-1] + input_array[i])
    return np.array(cdf, dtype=np.float)

def get_cdf_from_bins(input_array, input_bins):
    sort_idx = np.argsort(input_array)
    sorted_input = input_array[sort_idx]
    #histogram of the values that occur ignoring the frequency
    h,bins = np.histogram(sorted_input, bins = input_bins)
    center_vals = 0.5*np.diff(bins) + bins[:-1]
    cdf = get_cdf(h)
    return center_vals, cdf

def get_hist_from_bins(input_array, input_bins):
    sort_idx = np.argsort(input_array)
    sorted_input = input_array[sort_idx]
    #histogram of the values that occur ignoring the frequency
    h,bins = np.histogram(sorted_input, bins = input_bins)
    center_vals = 0.5*np.diff(bins) + bins[:-1]
    return center_vals, h

if __name__ == "__main__":
    # load the grid
    proj_folder = dirname(dirname(dirname(os.path.realpath(__file__))))
    grid_path = os.path.join(proj_folder, 'data', 'grids', '64_64_nonperiodic.pkl')
    with open(grid_path, 'rb') as input:
        grid = pickle.load(input)
    # path to input training data
    # input_path = os.path.join(proj_folder, 'data', 'data_64_nonperiodic.npz')
    # input_file = np.load(input_path)
    # input_perm = input_file['X']
    # input_pressure = input_file['Y']
    # path to the network prediction file
    # result_path = os.path.join(proj_folder, 'results', 'inception_2', 'models',
    #                            'best_validation_model.npz')
    result_path = os.path.join(proj_folder, 'results/inception4_nodrop',
                               'best_validation_model.npz')
    result_file = np.load(result_path)
    # result_perm = result_file['perm']
    # result_pressure = result_file['pres_train']
    result_perm = result_file['perm']
    result_pressure = result_file['best_pres']
    result_real_pressure = result_file['pressure']
    u_model, v_model = velocity_from_pressure(result_perm, result_pressure, grid)
    u_input, v_input = velocity_from_pressure(result_perm, result_real_pressure, grid)
    # print(u_input)
    # print(u_model)
    vmin = min(np.amin(u_input), np.amin(u_model))
    vmax = max(np.amax(u_input), np.amax(u_model))
    # bins = np.linspace(vmin, vmax, 100)
    bins = np.linspace(-0.2, 0.2, 100)
    centers, input_cdf = get_cdf_from_bins(u_input, bins)
    _, model_cdf = get_cdf_from_bins(u_model, bins)
    fig, ax = plt.subplots(1,1)
    ax.hold(True)
    ax.plot(centers, input_cdf, label='True')
    ax.plot(centers, model_cdf, label='model')
    ax.legend()
    fig.savefig('/home/amirhossein/Desktop/temp.png', format='png')
    plt.close(fig)
    centers, input_cdf = get_hist_from_bins(u_input, bins)
    _, model_cdf = get_hist_from_bins(u_model, bins)
    fig, ax = plt.subplots(1, 1)
    ax.hold(True)
    ax.plot(centers, input_cdf, label='True')
    ax.plot(centers, model_cdf, label='model')
    ax.legend()
    fig.savefig('/home/amirhossein/Desktop/temp2.png', format='png')
    plt.close(fig)


