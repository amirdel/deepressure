import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
from model import Model
import os as os
import matplotlib.pyplot as plt

class NNModel(Model):
    def add_placeholders(self):
        """Adds placeholder variables to tensorflow computational graph.
        Tensorflow uses placeholder variables to represent locations in a
        computational graph where data is inserted.  These placeholders are used as
        inputs by the rest of the model building and will be fed data during
        training.
        See for more information:
        https://www.tensorflow.org/versions/r0.7/api_docs/python/io_ops.html#placeholders
        """

        self.perm_placeholder   = tf.placeholder(tf.float32,   (None, self.config.nx,self.config.nx,1), name = "perm")
        self.pressure_placeholder  = tf.placeholder(tf.float32,   (None, self.config.nx,self.config.nx,1), name = "pressure")
        self.Div_U_operator_placeholder = tf.sparse_placeholder(tf.float32,   (None, self.config.nfaces, self.config.nx*self.config.nx), name = "U_face_operator")
        self.is_training = tf.placeholder(tf.bool)

    def create_feed_dict(self, perm_batch, Div_U_operator_batch, is_training, U_pressure_batch = None):
        """Creates the feed_dict for one step of training.
        A feed_dict takes the form of:
        feed_dict = {
                <placeholder>: <tensor of values to be passed for placeholder>,
                ....
        }
        If labels_batch is None, then no labels are added to feed_dict.
        Hint: The keys for the feed_dict should be a subset of the placeholder
                    tensors created in add_placeholders.
        Args:
            inputs_batch: A batch of input data.
            labels_batch: A batch of label data.
        Returns:
            feed_dict: The feed dictionary mapping from placeholders to values.

        """
        config = self.config
        n_batch = len(perm_batch)
        indices = np.array([[]])
        values = np.array([])
        for i in range(n_batch):
            batch_num = i*np.ones((len(Div_U_operator_batch[i].col), 1))
            row  = Div_U_operator_batch[i].row.reshape(len(batch_num), 1)
            col = Div_U_operator_batch[i].col.reshape(len(batch_num), 1)
            indices_batch = np.concatenate((batch_num, row, col),axis=1)
            if i == 0:
                indices = indices_batch
                values = Div_U_operator_batch[i].data
            else:
                indices = np.concatenate((indices, indices_batch))
                values = np.concatenate((values, Div_U_operator_batch[i].data))

        shape = np.array([n_batch, config.nfaces,config.nx*config.nx], dtype=np.int64)
        indices = indices.astype(int)
        if U_pressure_batch is not None:
            feed_dict = {
                self.perm_placeholder: perm_batch,
                self.Div_U_operator_placeholder: (indices,values,shape),
                self.pressure_placeholder: U_pressure_batch,
                self.is_training: is_training,
            }
        else:
            feed_dict = {
                self.perm_placeholder: perm_batch,
                self.Div_U_operator_placeholder: (indices,values,shape),
                self.is_training: is_training,
            }
        ### END YOUR CODE
        return feed_dict

    def inception(self, layer_input):
        relu = tf.nn.relu
        conv3a_2 = tf.layers.conv2d(layer_input, filters=192, kernel_size=[1, 1], kernel_initializer=tf.contrib.layers.xavier_initializer(), padding="same", activation=relu)
        conv3a_upscaled_2 = tf.image.resize_images(conv3a_2, [config.nx, config.nx])
        conv3b_2 = tf.layers.conv2d(layer_input, filters=128, kernel_size=[1, 1], kernel_initializer=tf.contrib.layers.xavier_initializer(), padding="same", activation=relu)
        conv4b_2 = tf.layers.conv2d(conv3b_2 ,filters=192,kernel_size=[3,3],kernel_initializer=tf.contrib.layers.xavier_initializer(), padding="same",activation=relu)
        conv4b_upscaled_2 = tf.image.resize_images(conv4b_2, [config.nx, config.nx])
        conv3c_2 = tf.layers.conv2d(layer_input, filters=128, kernel_size=[1, 1], kernel_initializer=tf.contrib.layers.xavier_initializer(), padding="same", activation=relu)
        conv4c_2 = tf.layers.conv2d(conv3c_2 ,filters=128,kernel_size=[5,5],kernel_initializer=tf.contrib.layers.xavier_initializer(), padding="same",activation=relu)
        conv4c_upscaled_2 = tf.image.resize_images(conv4c_2, [config.nx, config.nx])
        pool3_2 = tf.layers.max_pooling2d(inputs=layer_input, pool_size=[3, 3], strides=1, padding ='same')
        pool3_conv1_2 = tf.layers.conv2d(pool3_2, filters=128,kernel_size=[1,1],kernel_initializer=tf.contrib.layers.xavier_initializer(), padding="same",activation=relu)
        conv2_pool3_upscaled_2 = tf.image.resize_images(pool3_conv1_2, [config.nx, config.nx])
        layer_output = relu(tf.concat([layer_input, conv3a_upscaled_2, conv4b_upscaled_2, conv4c_upscaled_2, conv2_pool3_upscaled_2], axis=3))
        return layer_output

    def add_prediction_op(self):
        """Implements the core of the model that transforms a batch of input data into predictions.
        Returns:
            pred: A tensor of shape (batch_size, n_classes)
        """
        relu = tf.nn.relu
        xavier = tf.contrib.layers.xavier_initializer()
        config = self.config

        conv1_7x7_s2 = tf.layers.conv2d(self.perm_placeholder, filters=96,kernel_size=[7,7],kernel_initializer=xavier, padding="same",activation=relu)
        pool1_3x3_s2 = tf.layers.max_pooling2d(inputs=conv1_7x7_s2, pool_size=[3,3], strides=2, padding = 'same')
        pool1_norm1 = tf.nn.lrn(pool1_3x3_s2)
        conv2_3x3_reduce = tf.layers.conv2d(pool1_norm1, filters=128,kernel_size=[1,1],kernel_initializer=xavier, padding="same",activation=relu)
        conv2_3x3 = tf.layers.conv2d(conv2_3x3_reduce ,filters=128,kernel_size=[3,3],kernel_initializer=xavier, padding="same",activation=relu)
        conv2_norm2 = tf.nn.lrn(conv2_3x3)
        pool2_3x3_s2 = tf.layers.max_pooling2d(inputs=conv2_norm2, pool_size=[3,3], strides=2, padding = 'same')
        
        conv3a = tf.layers.conv2d(pool2_3x3_s2, filters=128,kernel_size=[1,1],kernel_initializer=xavier, padding="same",activation=relu)
        conv3a_upscaled = tf.image.resize_images(conv3a, [config.nx, config.nx])
        conv3b = tf.layers.conv2d(pool2_3x3_s2 ,filters=128,kernel_size=[1,1],kernel_initializer=xavier, padding="same",activation=relu)
        conv4b = tf.layers.conv2d(conv3b ,filters=128,kernel_size=[3,3],kernel_initializer=xavier, padding="same",activation=relu)
        conv4b_upscaled = tf.image.resize_images(conv4b, [config.nx, config.nx])
        conv3c = tf.layers.conv2d(pool2_3x3_s2, filters=128,kernel_size=[1,1],kernel_initializer=xavier, padding="same",activation=relu)
        conv4c = tf.layers.conv2d(conv3c ,filters=128,kernel_size=[5,5],kernel_initializer=xavier, padding="same",activation=relu)
        conv4c_upscaled = tf.image.resize_images(conv4c, [config.nx, config.nx])
        pool3 = tf.layers.max_pooling2d(inputs=pool2_3x3_s2, pool_size=[3,3], strides=1, padding = 'same')
        pool3_conv1 = tf.layers.conv2d(pool3, filters=128,kernel_size=[1,1],kernel_initializer=xavier, padding="same",activation=relu)
        conv2_pool3_upscaled = tf.image.resize_images(pool3_conv1, [config.nx, config.nx])
        inception1 = relu(tf.concat([conv1_7x7_s2,conv3a_upscaled,conv4b_upscaled,conv4c_upscaled,conv2_pool3_upscaled], axis=3))

        inception2 = self.inception(inception1)
        last_inception = self.inception(inception2)

        inception_final_conv1 = tf.layers.conv2d(last_inception ,filters=192,kernel_size=[3,3],kernel_initializer=xavier, padding="same",activation=relu)
        inception_final_conv2 = tf.layers.conv2d(inception_final_conv1 ,filters=256,kernel_size=[1,1],kernel_initializer=xavier, padding="same",activation=relu)

        pressure = tf.layers.conv2d(inputs=inception_final_conv2, filters=1,kernel_size=[1,1],kernel_initializer=xavier, padding="same")
        pres_flat = tf.reshape(pressure,[-1,config.nx*config.nx,1])*config.max_val + config.mean_val
        dense_operator = tf.sparse_tensor_to_dense(tf.sparse_reorder(self.Div_U_operator_placeholder))
        Divergence = tf.matmul(dense_operator, pres_flat)
        Divergence = tf.reshape(Divergence,[-1,config.nfaces])
        return Divergence, pressure

    def test_matrix_op(self, sess, input_set):
        # pass the dev set and calculate the face velocity from the input pressure and operator
        input_p_flat = tf.reshape(self.pressure_placeholder, [-1,config.nx*config.nx,1])
        div_u = tf.matmul(tf.sparse_tensor_to_dense(tf.sparse_reorder(self.Div_U_operator_placeholder)),input_p_flat)
        div_u_reshaped = tf.reshape(div_u,[-1,config.nfaces])
        perm_dev, Div_U_operator_dev, U_pressure_dev =  zip(*input_set)
        feed = self.create_feed_dict(perm_dev, Div_U_operator_dev, True, U_pressure_dev)
        diff = sess.run(div_u_reshaped, feed_dict=feed)
        return diff

    def add_loss_op(self, pred_divergence, pred_pressure):
        """Adds Ops for the loss function to the computational graph.
        Args:
            pred: A tensor of shape (batch_size, n_classes)
        Returns:
            loss: A 0-d tensor (scalar) output
        """
        v_weight = config.weight
        loss = tf.nn.l2_loss(pred_pressure - self.pressure_placeholder) + \
               v_weight * tf.nn.l2_loss(pred_divergence) #
        return loss

    def add_training_op(self, loss):
        """Sets up the training Ops.
        Creates an optimizer and applies the gradients to all trainable variables.
        The Op returned by this function is what must be passed to the
        sess.run() to train the model. See
        https://www.tensorflow.org/versions/r0.7/api_docs/python/train.html#Optimizer
        for more information.
        Args:
            loss: Loss tensor (a scalar).
        Returns:
            train_op: The Op for training.
        """
        train_op = tf.train.AdamOptimizer(self.config.lr).minimize(loss)
        return train_op

    def train_on_batch(self, sess, perm_batch, Div_U_operator_batch, U_pressure_batch):
        """Perform one step of gradient descent on the provided batch of data.
        Args:
            sess: tf.Session()
            input_batch: np.ndarray of shape (n_samples, n_features)
            labels_batch: np.ndarray of shape (n_samples, n_classes)
        Returns:
            loss: loss over the batch (a scalar)
        """
        feed = self.create_feed_dict(perm_batch , Div_U_operator_batch, True, U_pressure_batch)
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        return loss/(len(perm_batch))

    def predict_on_batch(self, sess, perm_batch, Div_U_operator_batch):
        """Make predictions for the provided batch of data
        Args:
            sess: tf.Session()
            input_batch: np.ndarray of shape (n_samples, n_features)
        Returns:
            predictions: np.ndarray of shape (n_samples, n_classes)
        """
        feed = self.create_feed_dict(perm_batch, Div_U_operator_batch, False)
        predictions, pres = sess.run([self.pred, self.pres], feed_dict=feed)
        # pres = sess.run(self.pres, feed_dict=feed)
        # u_diff = sess.run()
        return predictions, pres

    def fit(self, sess, train_set, dev_set):
        save_dir = self.config.model_save_dir
        save_file = os.path.join(save_dir, 'best_validation_model')
        best_dev = 1e9
        best_velocity = None
        best_pres = None

        for epoch in range(self.config.n_epochs):
            self.epoch_count = epoch
            print ("Epoch {:} out of {:}".format(epoch + 1, self.config.n_epochs))
            dev_score, pred , pres= self.run_epoch(sess, train_set, dev_set)
            self.save_loss_history(os.path.join(config.proj_folder, 'temp', config.model_name, 'pics'))
            if dev_score < best_dev:
                best_dev = dev_score
                best_velocity = pred
                best_pres = pres
                perm_dev, Div_U_operator_dev, U_pressure_dev = zip(*dev_set)
                np.savez(save_file,best_velocity=best_velocity, best_pres = best_pres, perm=perm_dev,
                         U_face_operator = Div_U_operator_dev, pressure=U_pressure_dev)
                print("new best norm found {:}".format(best_dev))
                save_path = self.saver.save(sess, self.config.save_model_path)
                print("Model saved in file: %s" % save_path)

    def run_epoch(self, sess, train_examples, dev_set):
        save_dir = self.config.model_save_dir
        latest_file = os.path.join(save_dir, 'latest_train_model')
        config = self.config
        num_train = len(train_examples)
        ind = np.arange(num_train)
        np.random.shuffle(ind)
        ind.astype(int)
        batchNum = 1
        num_batch = int(np.ceil(num_train/config.batch_size))
        for i in range(0, num_train,config.batch_size):
            batch_train = [ train_examples[i] for i in ind[i:i+config.batch_size]]
            perm_train, Div_U_operator_train, U_pressure_train =  zip(*batch_train)

            loss = self.train_on_batch(sess, perm_train, Div_U_operator_train, U_pressure_train)
            self.recordOutput(loss,batchNum)

            print("Loss for Batch {:} out of {:} is: {:}".format(batchNum,num_batch,loss))
            batchNum += 1

        pred_train ,pres_train= self.predict_on_batch(sess, perm_train, Div_U_operator_train)
        np.savez(latest_file, pred_train=pred_train,pres_train=pres_train,perm=perm_train,U_face_operator = Div_U_operator_train,
                 pressure=U_pressure_train)

        print("------Evaluating on dev set------")
        #batch over dev
        num_dev = len(dev_set)
        batchNum = 1
        ind_dev = np.arange(num_dev)
        num_batch = int(np.ceil(num_dev/config.batch_size))
        norms = []
        predAll = np.zeros((num_dev,self.config.nfaces))
        presAll = np.zeros((num_dev,self.config.nx,self.config.nx,1))
        for devInd in range(0, num_dev,config.batch_size):
            batch_dev = [ dev_set[i] for i in ind_dev[devInd:devInd+config.batch_size]]
            perm_dev, Div_U_operator_dev, U_pressure_dev =  zip(*batch_dev)
            pred, pres = self.predict_on_batch(sess,perm_dev, Div_U_operator_dev)
            # calculate norm of pressure
            for j in range(len(U_pressure_dev)):
                norms.append(np.sum(np.power(U_pressure_dev[j]-pres[j,:,:,:],2)))
            presAll[devInd:devInd+config.batch_size] = pres
            predAll[devInd:devInd+config.batch_size] = pred
            averageNorm = np.sum(norms)*0.5/len(norms)
            print("Norm for Batch {:} out of {:} is: {:}".format(batchNum,num_batch,averageNorm))
            batchNum += 1
        return np.sum(norms)*0.5/len(dev_set), predAll, presAll

    def recordOutput(self,loss,batchNum):
        self.loss_history.append(loss)
        self.iter_number.append(float(self.epoch_count)+np.float(batchNum*self.config.batch_size/np.float(self.config.n_train)))

    def save_loss_history(self, save_folder):
        np.savez(os.path.join(save_folder, 'loss'), loss=self.loss_history, iter_number=self.iter_number)
        fig, ax = plt.subplots(1,1)
        ax.plot(self.iter_number, self.loss_history)
        fig.savefig(os.path.join(save_folder, 'loss.png'), format='png')

    def build(self):
        self.add_placeholders()
        self.pred, self.pres = self.add_prediction_op()
        self.loss = self.add_loss_op(self.pred,self.pres)
        self.train_op = self.add_training_op(self.loss)
        self.saver = tf.train.Saver()
        self.loss_history = []
        self.iter_number = []

    def __init__(self, config):
        self.config = config
        self.build()

from os.path import dirname
class Config():
    weight = 0.5
    lr = 1e-4
    load = False
    n_epochs = 100
    kernel_size = 6
    batch_size = 25
    n_filters = 10
    dropout = 0.2
    model_name = 'inception_3_lr1e4_bigger'
    proj_folder = dirname(dirname(dirname(os.path.realpath(__file__))))
    model_save_dir = os.path.join(proj_folder, 'temp', model_name, 'models')
    save_model_path = os.path.join(model_save_dir, 'model_saved.ckpt')

if __name__ == "__main__":
    print("Started running")
    config = Config()
    proj_folder = dirname(dirname(dirname(os.path.realpath(__file__))))
    print(proj_folder)
    datFile = np.load(os.path.join(proj_folder,'data','data_64_nonperiodic.npz'))


    print("Making folders")
    proj_folder = dirname(dirname(dirname(os.path.realpath(__file__))))
    # path to save the training data file
    save_folder = os.path.join(proj_folder, 'temp', config.model_name, 'pics')
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    model_folder = os.path.join(proj_folder, 'temp', config.model_name, 'models')
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    X = datFile['X']
    Y_in = datFile['Y']
    Div_U_operator = datFile['Div_u_operator']
    config.nx = X.shape[1]
    # TODO: config.nfaces is actually ncells
    config.nfaces = config.nx**2
    # TODO: normalization removed
    config.mean_val = np.mean(Y_in)
    config.max_val = np.max(np.fabs(Y_in-config.mean_val))*(4/3)
    # config.mean_val = 0.0
    # config.max_val = 1.0
    # TODO: normalization removed
    Y = (Y_in-config.mean_val)/config.max_val
    # Y = Y_in
    n_train = int(X.shape[0]*0.8)
    n_dev = X.shape[0]-n_train

    train_set = []
    for i in range(n_train):
        train_set.append((X[i, :, :, :], Div_U_operator[i], Y[i, :, :, :]))

    dev_set = []
    for i in range(n_dev):
        dev_set.append((X[i+n_train, :, :, :], Div_U_operator[i + n_train], Y[i + n_train, :, :, :]))

    config.n_train = n_train
    with tf.Graph().as_default():
        print ("building Model")
        model = NNModel(config)
        
        with tf.Session() as session:
            if config.load:
                # saver object to save things
                print ("loading model")
                model.saver.restore(session, model.config.save_model_path)
            else:
                print ("initializing model from scratch")
                init = tf.global_variables_initializer()
                session.run(init)

            if n_train < 25:
                diff = model.test_matrix_op(session, train_set)
                print('sanity check:')
                print('mean diff: ', np.mean(diff))
                print('max diff: ', np.amax(np.abs(diff)))
                print('error for reproducing divergence: ', np.linalg.norm(diff))
                print(np.sum(np.array(np.abs(diff) > 1e-4, dtype=np.int)))
                
            print('Number of trainable parameters : ',np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
            model.fit(session, train_set, dev_set)
            model.save_loss_history(os.path.join(proj_folder, 'temp', config.model_name, 'pics'))


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

    for i in range(2):
        save_path = os.path.join(save_folder, 'p_last_epoch{:}.png'.format(i))
        side_by_side(x_mat, y_mat, p_pred[i, :, :, 0], pressure_actual[i, :, :, 0], save_path)
