
import numpy as np
import os as os
import matplotlib.pyplot as plt
import pickle as pickle
from deepres.simulator.periodic_field_functions import PeriodicPerturbations

proj_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(proj_folder)
# path to save the training data file
save_folder = os.path.join(proj_folder, 'temp')
save_path = os.path.join(save_folder, 'data.npz')
grid_path = os.path.join(proj_folder, 'sample_scripts', 'script_files', '100_100_periodic.pkl')

data = np.load(save_path)
X = data['X']
Y = data['Y']
face_vel_operator = data['U_face_operator']
face_vel_bias = data['U_face_fixed']


with open(grid_path, 'rb') as input:
    grid = pickle.load(input)
n_cells = grid.nr_p
dx, dy = grid.dx, grid.dy
nx = grid.m
gridx = grid.pores.x
gridy = grid.pores.y
x_mat = np.reshape(gridx, (nx,nx), order='F')
y_mat = np.reshape(gridy, (nx,nx), order='F')
PI = PeriodicPerturbations(grid, None, None)
# plot permeability
for i in range(3):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    perm_mat = X[i,:,:,0]
    p = ax.pcolormesh(x_mat, y_mat, perm_mat, cmap=plt.cm.coolwarm)
    # ax.set_aspect('equal', 'box')
    # plot the cell center velocity on top of perm
    div = face_vel_operator[i]
    pressure_vec = Y[i,:,:,0].reshape((n_cells))
    u_bias = face_vel_bias[i,:]
    u_face = div.dot(pressure_vec) + u_bias
    # get cell center velocity from face velocity
    u_cell, v_cell = PI.get_cell_velocity(u_face)
    u_mat = np.reshape(u_cell, (nx,nx), order='F')
    v_mat = np.reshape(v_cell, (nx,nx), order='F')
    f = 12
    ax.quiver(gridx[::f], gridy[::f], u_cell[::f], v_cell[::f], units='inches')
    # ax.streamplot(x_mat, y_mat, u_mat, v_mat, linewidth=1.0, color='w')
    ax.set_ybound([0,nx])
    ax.set_xbound([0,nx])
    ax.set_aspect('equal', 'box')
    cbar = fig.colorbar(p, fraction=0.046, pad=0.04)
    fig.savefig(os.path.join(save_folder, 'perm'+str(i)+'.png'), format='png')
    # plot pressure
    p_mat = Y[i,:,:,0]
    fig, ax = plt.subplots(1,1)
    p = ax.pcolormesh(x_mat, y_mat, p_mat, cmap=plt.cm.coolwarm)
    cbar = fig.colorbar(p, fraction=0.046, pad=0.04)
    ax.set_aspect('equal', 'box')
    fig.savefig(os.path.join(save_folder, 'pressure'+str(i)+'.png'), format='png')