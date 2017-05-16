import numpy as np
import os as os
import matplotlib.pyplot as plt
import pickle as pickle
from deepres.simulator.periodic_field_functions import PeriodicPerturbations


proj_folder = os.path.dirname(os.path.realpath(__file__))
# path to save the training data file
save_folder = os.path.join(proj_folder, 'temp')
save_path = os.path.join(save_folder, 'data.npz')
grid_path = os.path.join(proj_folder, 'sample_scripts', 'script_files', '100_100_periodic.pkl')

#data = np.load(save_path)
#X = data['X']
#Y = data['Y']

datFile =np.load('best_pred.npz')

Y_pred = datFile['best_pred']
X = datFile['perm_dev']
u_actual = datFile['U_face_dev']
u_pred = datFile['best_pred']


with open(grid_path, 'rb') as input:
    grid = pickle.load(input)
dx, dy = grid.dx, grid.dy
nx = grid.m
gridx = grid.pores.x
gridy = grid.pores.y
x_mat = np.reshape(gridx, (nx,nx), order='F')
y_mat = np.reshape(gridy, (nx,nx), order='F')
# plot permeability

PI = PeriodicPerturbations(grid, None, None)
for i in range(5):
	fig = plt.figure()
	ax = fig.add_subplot(1,2,1)
	perm_mat = X[i,:,:,0]
	p = ax.pcolormesh(x_mat, y_mat, perm_mat, cmap=plt.cm.coolwarm)
	ax.set_aspect('equal', 'box')
	u_cell, v_cell = PI.get_cell_velocity(u_actual[i,:])
	u_mat = np.reshape(u_cell, (nx,nx), order='F')
	v_mat = np.reshape(v_cell, (nx,nx), order='F')
	f = 12
	ax.quiver(gridx[::f], gridy[::f], u_cell[::f], v_cell[::f], units='inches')

	# f = 12
	# ax.quiver(gridx[::f], gridy[::f], u[::f], v[::f], units='inches')
	# ax.streamplot(x_mat, y_mat, u_mat, v_mat, linewidth=1.0, color='w')
	# ax.quiver(gridx, gridy, u, v, units='width')
	ax.set_ybound([0,nx])
	ax.set_xbound([0,nx])
	ax.set_aspect('equal', 'box')
	cbar = fig.colorbar(p, fraction=0.046, pad=0.04)

	ax = fig.add_subplot(1,2,2)
	perm_mat = X[i,:,:,0]
	p = ax.pcolormesh(x_mat, y_mat, perm_mat, cmap=plt.cm.coolwarm)
	ax.set_aspect('equal', 'box')
	u_cell, v_cell = PI.get_cell_velocity(u_pred[i,:])
	u_mat = np.reshape(u_cell, (nx,nx), order='F')
	v_mat = np.reshape(v_cell, (nx,nx), order='F')
	f = 12
	ax.quiver(gridx[::f], gridy[::f], u_cell[::f], v_cell[::f], units='inches')
	# f = 12
	# ax.quiver(gridx[::f], gridy[::f], u[::f], v[::f], units='inches')
	# ax.streamplot(x_mat, y_mat, u_mat, v_mat, linewidth=1.0, color='w')
	# ax.quiver(gridx, gridy, u, v, units='width')
	ax.set_ybound([0,nx])
	ax.set_xbound([0,nx])
	ax.set_aspect('equal', 'box')
	cbar = fig.colorbar(p, fraction=0.046, pad=0.04)

	fig.savefig(os.path.join(save_folder, 'perm{:}.png'.format(i)), format='png')


	# plot pressure
	# p_mat = Y_pred[i,:,:,0]
	# fig = plt.figure()
	# ax = fig.add_subplot(1,2,1)
	# p = ax.pcolormesh(x_mat, y_mat, p_mat, cmap=plt.cm.coolwarm,vmin = -1,vmax = 1)
	# cbar = fig.colorbar(p, fraction=0.046, pad=0.04,ticks = [-1,0,1])

	# ax.set_aspect('equal', 'box')

	# p_mat = Y_actual[i,:,:,0]
	# ax = fig.add_subplot(1,2,2)
	# p = ax.pcolormesh(x_mat, y_mat, p_mat, cmap=plt.cm.coolwarm,vmin = -1,vmax = 1)
	# cbar = fig.colorbar(p, fraction=0.046, pad=0.04,ticks = [-1,0,1])

	# ax.set_aspect('equal', 'box')
	# fig.savefig(os.path.join(save_folder, 'pressure_both{:}.png'.format(i)), format='png')