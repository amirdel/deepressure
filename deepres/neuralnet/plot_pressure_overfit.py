import numpy as np
import os as os
import pickle as pickle
from deepres.plotting.side_by_side import side_by_side
from os.path import dirname

print('comparing the pressure solutions...')
proj_folder = dirname(dirname(dirname(os.path.realpath(__file__))))
# path to save the training data file
save_folder = os.path.join(proj_folder, 'temp', 'p_overfit', 'pics')
if not os.path.exists(save_folder):
    os.mkdir(save_folder)
data_path = os.path.join(proj_folder, 'data', 'data_500.npz')
# grid_path = os.path.join(proj_folder, 'sample_scripts', 'script_files', '100_100_periodic.pkl')
grid_path = os.path.join(proj_folder, 'data', 'grids', '64_64_periodic.pkl')
model_folder = os.path.join(proj_folder, 'temp', 'p_overfit', 'models')
# load model
modelfile = np.load(os.path.join(model_folder, 'latest_train_model.npz'))

X = modelfile['perm']
pressure_actual = modelfile['pressure']
p_pred = modelfile['best_pred']

print('shape of model: ', p_pred.shape)
with open(grid_path, 'rb') as input:
    grid = pickle.load(input)
dx, dy = grid.dx, grid.dy
nx = grid.m
gridx = grid.pores.x
gridy = grid.pores.y
x_mat = np.reshape(gridx, (nx, nx), order='F')
y_mat = np.reshape(gridy, (nx, nx), order='F')

for i in range(2):
    save_path = os.path.join(save_folder, 'pressure_both{:}.png'.format(i))
    side_by_side(x_mat, y_mat, p_pred[i,:,:,0], pressure_actual[i,:,:,0], save_path)