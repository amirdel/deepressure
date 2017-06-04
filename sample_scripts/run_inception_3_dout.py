from os.path import dirname
import os as os
import tensorflow as tf
import numpy as np
from deepres.neuralnet.inception2 import InceptionTwo

class Config():
    weight = 8.0
    tv_weight = 6.0	
    lr = 2e-4
    init_lr = lr
    lr_decay = 0.96
    dropout = 0.2
    n_inception = 3
    load = False
    n_epochs = 100
    kernel_size = 6
    batch_size = 25
    model_name = 'inception3_dout'
    proj_folder = dirname(dirname(os.path.realpath(__file__)))
    result_root = os.path.join(proj_folder, 'results')
    model_save_dir = os.path.join(result_root, model_name, 'models')
    plot_dir = os.path.join(result_root, model_name, 'pics')
    save_model_path = os.path.join(model_save_dir, 'model_saved.ckpt')

print("Started running")
config = Config()
proj_folder = config.proj_folder
print(proj_folder)
print('learning_rate: ', config.lr)
print('learning_rate decay: ', config.lr_decay)
print('number of layers: ', config.n_inception)
print('load: ', config.load)
print('div weight: ', config.weight)
print('tv weight: ', config.tv_weight)
print('batch size: ', config.batch_size)
print('dout: ', config.dropout)
datFile = np.load(os.path.join(proj_folder, 'data', 'data_64_nonperiodic_6000.npz'))
#datFile = np.load(os.path.join(proj_folder, 'data', 'data_64_nonperiodic.npz'))

print("Making folders")
# results root folder
result_root = config.result_root
case_folder = os.path.join(result_root, config.model_name)
save_folder = os.path.join(case_folder, 'pics')
model_folder = os.path.join(case_folder, 'models')
for folder in [result_root, case_folder, save_folder, model_folder]:
    if not os.path.exists(folder):
        os.mkdir(folder)
### setting number of training and validation
X = datFile['X']
Y_in = datFile['Y']
Div_U_operator = datFile['Div_u_operator']
config.nx = X.shape[1]
# TODO: config.nfaces is actually ncells
config.nfaces = config.nx ** 2
config.mean_val = np.mean(Y_in)
config.max_val = np.max(np.fabs(Y_in - config.mean_val)) * (4 / 3)
Y = (Y_in - config.mean_val) / config.max_val

n_examples = X.shape[0]
n_train = int(0.8*n_examples)
n_dev = n_examples - n_train
print('n_train: ', n_train)
print('n_dev: ', n_dev)

train_set = []
for i in range(n_train):
    train_set.append((X[i, :, :, :], Div_U_operator[i], Y[i, :, :, :]))

dev_set = []
for i in range(n_dev):
    dev_set.append((X[i + n_train, :, :, :], Div_U_operator[i + n_train], Y[i + n_train, :, :, :]))

config.n_train = n_train
with tf.Graph().as_default():
    print("building Model")
    model = InceptionTwo(config)

    with tf.Session() as session:
        if config.load:
            # saver object to save things
            print("loading model")
            model.saver.restore(session, model.config.save_model_path)
        else:
            print("initializing model from scratch")
            init = tf.global_variables_initializer()
            session.run(init)

        # if n_train < 200:
        #     diff = model.test_matrix_op(session, train_set)
        #     print('sanity check:')
        #     print('mean diff: ', np.mean(diff))
        #     print('max diff: ', np.amax(np.abs(diff)))
        #     print('error for reproducing divergence: ', np.linalg.norm(diff))
        #     print(np.sum(np.array(np.abs(diff) > 1e-4, dtype=np.int)))

        print('Number of trainable parameters : ',
              np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
        model.fit(session, train_set, dev_set, reduce_every=1)
        model.save_loss_history(config.plot_dir)

################################################################################################
# plot the results
import numpy as np
import os as os
import pickle as pickle
from deepres.plotting.side_by_side import side_by_side
from os.path import dirname

print('comparing the pressure solutions...')
# grid_path = os.path.join(proj_folder, 'sample_scripts', 'script_files', '100_100_periodic.pkl')
grid_path = os.path.join(proj_folder, 'data', 'grids', '64_64_periodic.pkl')
# load model
modelfile = np.load(os.path.join(model_folder, 'latest_train_model.npz'))

X = modelfile['perm']
pressure_actual = modelfile['pressure']
p_pred = modelfile['pres_train']

print('shape of model: ', p_pred.shape)
with open(grid_path, 'rb') as input:
    grid = pickle.load(input)
dx, dy = grid.dx, grid.dy
nx = grid.m
gridx = grid.pores.x
gridy = grid.pores.y
x_mat = np.reshape(gridx, (nx, nx))
y_mat = np.reshape(gridy, (nx, nx))

for i in range(10):
    save_path = os.path.join(save_folder, 'p_last_epoch{:}.png'.format(i))
    side_by_side(x_mat, y_mat, p_pred[i, :, :, 0], pressure_actual[i, :, :, 0], save_path)
