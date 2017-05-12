import numpy as np
import os as os
import matplotlib.pyplot as plt
import pickle as pickle

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
print(Y_pred)
X = datFile['input_dev']
Y_actual = datFile['labels_dev']


with open(grid_path, 'rb') as input:
    grid = pickle.load(input)
dx, dy = grid.dx, grid.dy
nx = grid.m
gridx = grid.pores.x
gridy = grid.pores.y
x_mat = np.reshape(gridx, (nx,nx), order='F')
y_mat = np.reshape(gridy, (nx,nx), order='F')
# plot permeability
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
perm_mat = X[1,:,:,0]
p = ax.pcolormesh(x_mat, y_mat, perm_mat, cmap=plt.cm.coolwarm)
ax.set_aspect('equal', 'box')
# f = 12
# ax.quiver(gridx[::f], gridy[::f], u[::f], v[::f], units='inches')
# ax.streamplot(x_mat, y_mat, u_mat, v_mat, linewidth=1.0, color='w')
# ax.quiver(gridx, gridy, u, v, units='width')
ax.set_ybound([0,nx])
ax.set_xbound([0,nx])
ax.set_aspect('equal', 'box')
cbar = fig.colorbar(p, fraction=0.046, pad=0.04)
fig.savefig(os.path.join(save_folder, 'perm.png'), format='png')
# plot pressure
p_mat = Y_pred[1,:,:,0]
fig, ax = plt.subplots(1,1)
p = ax.pcolormesh(x_mat, y_mat, p_mat, cmap=plt.cm.coolwarm)
cbar = fig.colorbar(p, fraction=0.046, pad=0.04)

ax.set_aspect('equal', 'box')
fig.savefig(os.path.join(save_folder, 'pressure_pred.png'), format='png')

p_mat = Y_actual[1,:,:,0]
fig, ax = plt.subplots(1,1)
p = ax.pcolormesh(x_mat, y_mat, p_mat, cmap=plt.cm.coolwarm)
cbar = fig.colorbar(p, fraction=0.046, pad=0.04)

ax.set_aspect('equal', 'box')
fig.savefig(os.path.join(save_folder, 'pressure_actual.png'), format='png')