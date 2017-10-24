
import numpy as np
import os as os
import matplotlib.pyplot as plt
import pickle as pickle
# from deepres.simulator.periodic_field_functions import PeriodicPerturbations
from deepres.simulator.operators import face_velocity_operator_nonperiodic, get_cell_velocity, divergence_operator

proj_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(proj_folder)
# path to save the training data file
save_folder = os.path.join(proj_folder, 'temp')
# save_path = os.path.join(save_folder, 'data.npz')
# grid_path = os.path.join(proj_folder, 'sample_scripts', 'script_files', '100_100_periodic.pkl')

save_path = os.path.join(proj_folder, 'data', 'data_64_nonperiodic.npz')
grid_path = os.path.join(proj_folder, 'data', 'grids', '64_64_nonperiodic.pkl')


data = np.load(save_path)
X = data['X']
Y = data['Y']
face_vel_operator = data['U_face_operator']
# face_vel_bias = data['U_face_fixed']


with open(grid_path, 'rb') as input:
    grid = pickle.load(input)
# get the divergence operator
div_operator = divergence_operator(grid)
n_cells = grid.nr_p
dx, dy = grid.dx, grid.dy
nx = grid.m
gridx = grid.pores.x
gridy = grid.pores.y
# x_mat = np.reshape(gridx, (nx,nx), order='F')
# y_mat = np.reshape(gridy, (nx,nx), order='F')
x_mat = np.reshape(gridx, (nx,nx))
y_mat = np.reshape(gridy, (nx,nx))
# plot permeability
for i in range(3):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    perm_mat = X[i,:,:,0]
    # cm = plt.cm.RdYlBu_r
    cm = plt.cm.coolwarm
    p = ax.pcolormesh(x_mat, y_mat, perm_mat, cmap=cm, alpha=1.0)
    # ax.set_aspect('equal', 'box')
    # plot the cell center velocity on top of perm
    face_v_operator = face_vel_operator[i]
    pressure_vec = Y[i,:,:,0].reshape((n_cells))
    # u_bias = face_vel_bias[i,:]
    u_face = face_v_operator.dot(pressure_vec) #+ u_bias
    print('mean velocity: ', np.mean(u_face))
    # calculate the divergence
    div = div_operator.dot(u_face)
    print('mean divergence: ', np.mean(div))
    # get cell center velocity from face velocity
    u_cell, v_cell = get_cell_velocity(grid, u_face)
    # u_mat = np.reshape(u_cell, (nx,nx), order='F')
    # v_mat = np.reshape(v_cell, (nx,nx), order='F')
    # ax.streamplot(x_mat, y_mat, u_mat, v_mat, linewidth=1.0, color='w')
    ax.set_ybound([0,nx])
    ax.set_xbound([0,nx])
    ax.set_aspect('equal', 'box')
    cbar = fig.colorbar(p, fraction=0.046, pad=0.04)
    u_mat = np.reshape(u_cell, (nx, nx))
    v_mat = np.reshape(v_cell, (nx, nx))
    f = 12
    # ax.quiver(gridx[::f], gridy[::f], u_cell[::f], v_cell[::f], scale=2**.5, units='y')
    # ax.quiver(gridx[::f], gridy[::f], u_cell[::f], v_cell[::f], units='inches')
    fig.savefig(os.path.join(save_folder, 'perm'+str(i)+'.png'), format='png')
    # plot pressure
    p_mat = Y[i,:,:,0]
    fig, ax = plt.subplots(1,1)
    p = ax.pcolormesh(x_mat, y_mat, p_mat, cmap=plt.cm.coolwarm)
    cbar = fig.colorbar(p, fraction=0.046, pad=0.04)
    ax.set_aspect('equal', 'box')
    fig.savefig(os.path.join(save_folder, 'pressure'+str(i)+'.png'), format='png')
    # plot the divergence
    fig, ax = plt.subplots(1,1)
    p = ax.pcolormesh(x_mat, y_mat, np.reshape(div, (nx,nx)), cmap=plt.cm.coolwarm)
    cbar = fig.colorbar(p, fraction=0.046, pad=0.04)
    ax.set_aspect('equal', 'box')
    fig.savefig(os.path.join(save_folder, 'divergence' + str(i) + '.png'), format='png')