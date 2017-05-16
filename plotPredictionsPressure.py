import numpy as np
import os as os
import matplotlib.pyplot as plt
import pickle as pickle
from matplotlib import gridspec
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from deepres.simulator.periodic_field_functions import PeriodicPerturbations


proj_folder = os.path.dirname(os.path.realpath(__file__))
# path to save the training data file
save_folder = os.path.join(proj_folder, 'temp')
save_path = os.path.join(save_folder, 'data.npz')
grid_path = os.path.join(proj_folder, 'sample_scripts', 'script_files', '100_100_periodic.pkl')

#data = np.load(save_path)
#X = data['X']
#Y = data['Y']

datFile =np.load('best_pred_pressure.npz')

X = datFile['perm']
pressure_actual = datFile['y_pressure']
p_pred = datFile['best_pred']


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
	u_actual_mat = np.reshape(u_cell, (nx,nx), order='F')
	v_actual_mat = np.reshape(v_cell, (nx,nx), order='F')
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
	u_pred_mat = np.reshape(u_cell, (nx,nx), order='F')
	v_pred_mat = np.reshape(v_cell, (nx,nx), order='F')
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

	fig = plt.figure(figsize=(9,4))

	gs = gridspec.GridSpec(1, 2)
	gs.update(left=0.1, right=0.86, hspace=0.1)
	gs1 = gridspec.GridSpec(1, 1)
	gs1.update(left=0.89, right=0.92)
	ax = plt.subplot(gs[0])
	ax2 = plt.subplot(gs[1])
	ax_cmap = plt.subplot(gs1[0])
	
	cmap = plt.cm.afmhot_r
	
	p = ax.pcolormesh(x_mat, y_mat, u_pred_mat, cmap=plt.cm.coolwarm,vmin = 0,vmax =0.5 )

	ax.set_aspect('equal', 'box')

	p2 = ax2.pcolormesh(x_mat, y_mat, u_actual_mat, cmap=plt.cm.coolwarm ,vmin = 0, vmax = 0.5)
	cbar = fig.colorbar(p, ax_cmap, orientation='vertical')
	
	ax.set_aspect('equal', 'box')
	fig.savefig(os.path.join(save_folder, 'u_both{:}.png'.format(i)), format='png')
	
	fig = plt.figure(figsize=(9,4))

	gs = gridspec.GridSpec(1, 2)
	gs.update(left=0.1, right=0.86, hspace=0.1)
	gs1 = gridspec.GridSpec(1, 1)
	gs1.update(left=0.89, right=0.92)
	ax = plt.subplot(gs[0])
	ax2 = plt.subplot(gs[1])
	ax_cmap = plt.subplot(gs1[0])
	
	cmap = plt.cm.afmhot_r
	
	p = ax.pcolormesh(x_mat, y_mat, v_pred_mat, cmap=plt.cm.coolwarm,vmin = -0.15,vmax =0.15 )

	ax.set_aspect('equal', 'box')

	p2 = ax2.pcolormesh(x_mat, y_mat, v_actual_mat, cmap=plt.cm.coolwarm ,vmin = -0.15, vmax = 0.15)
	cbar = fig.colorbar(p2, ax_cmap, orientation='vertical')
	
	ax.set_aspect('equal', 'box')
	fig.savefig(os.path.join(save_folder, 'v_both{:}.png'.format(i)), format='png')
	
	# def compare_trans_mat_vtheta(trans_matrix_1, trans_matrix_2, figure_save_folder, prefix, refsize = 16):

	# fontsize = refsize * 0.8
	# fmt = 'pdf'
	# fig = plt.figure(figsize=(9,4))

	# gs = gridspec.GridSpec(1, 2)
	# gs.update(left=0.1, right=0.86, hspace=0.1)
	# gs1 = gridspec.GridSpec(1, 1)
	# gs1.update(left=0.89, right=0.92)
	# ax = plt.subplot(gs[0])
	# ax2 = plt.subplot(gs[1])
	# ax_cmap = plt.subplot(gs1[0])
	# cmap = plt.cm.afmhot_r
	# max_prob = 1.0
	# # print max_prob
	# p = ax.pcolor(np.sqrt(trans_matrix_1), norm=colors.SymLogNorm(linthresh=0.03, linscale=0.03,vmin=0, vmax=max_prob),
	# 	cmap=cmap, linewidth=0, rasterized=True)
	# p2 = ax2.pcolor(np.sqrt(trans_matrix_2), norm=colors.SymLogNorm(linthresh=0.03, linscale=0.03,vmin=0, vmax=max_prob),
	# 	cmap=cmap, linewidth=0, rasterized=True)

	# cbar = fig.colorbar(p, ax_cmap, orientation='vertical')
	# tick_array = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
	# cbar.set_ticks(tick_array)
	# # cbar.set_ticklabels()
	# lp  = 1
	# ax.set_xlabel('previous class ' + r'($v_n$)', labelpad=lp)
	# ax.set_ylabel('next class '+ r'($v_{n+1}$)', labelpad=lp)
	# ax2.set_xlabel('previous class '+ r'($\theta_n$)', labelpad=lp)
	# ax2.set_ylabel('next class '+ r'($\theta_{n+1}$)', labelpad=lp)
	# mat_size = trans_matrix_1.shape[1]
	# ax_bound = [0, mat_size]
	# ax.set_xbound(ax_bound)
	# ax.set_ybound(ax_bound)
	# mat_size = trans_matrix_2.shape[1]
	# ax_bound = [0, mat_size]
	# ax2.set_xbound(ax_bound)
	# ax2.set_ybound(ax_bound)
	# fig_name = prefix +'.'+fmt
	# file_name = os.path.join(figure_save_folder, fig_name)
	# fig.savefig(file_name, format=fmt)
	# plt.close(fig)