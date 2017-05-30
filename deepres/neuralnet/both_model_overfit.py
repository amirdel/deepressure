import tensorflow as tf
import numpy as np
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
        self.U_face_operator_placeholder = tf.sparse_placeholder(tf.float32,   (None, self.config.nfaces,self.config.nx*self.config.nx), name = "U_face_operator")
        self.U_face_fixed_placeholder = tf.placeholder(tf.float32,   (None, self.config.nfaces), name = "U_face_fixed")
        self.U_face_placeholder = tf.placeholder(tf.float32,   (None, self.config.nfaces), name = "U_face")

        self.is_training = tf.placeholder(tf.bool)

    def create_feed_dict(self, perm_batch,U_face_fixed_batch,U_face_operator_batch, is_training,U_pressure_batch = None,U_face_batch=None):
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
            batch_num = i*np.ones((len(U_face_operator_batch[i].col),1))
            row  = U_face_operator_batch[i].row.reshape(len(batch_num),1)
            col = U_face_operator_batch[i].col.reshape(len(batch_num),1)
            indices_batch = np.concatenate((batch_num, row, col),axis=1)
            if i == 0:
                indices = indices_batch
                values = U_face_operator_batch[i].data
            else:
                indices = np.concatenate((indices, indices_batch))
                values = np.concatenate((values, U_face_operator_batch[i].data))

        shape = np.array([n_batch, config.nfaces,config.nx*config.nx], dtype=np.int64)
        indices = indices.astype(int)
        if U_face_batch is not None:
            feed_dict = {
                self.perm_placeholder: perm_batch,
                self.U_face_fixed_placeholder: U_face_fixed_batch,
                self.U_face_operator_placeholder: (indices,values,shape),
                self.U_face_placeholder: U_face_batch,
                self.pressure_placeholder: U_pressure_batch,
                self.is_training: is_training,
            }
        else:
            feed_dict = {
                self.perm_placeholder: perm_batch,
                self.U_face_fixed_placeholder: U_face_fixed_batch,
                self.U_face_operator_placeholder: (indices,values,shape),
                self.is_training: is_training,
            }
        ### END YOUR CODE
        return feed_dict

    def add_prediction_op(self):
        """Implements the core of the model that transforms a batch of input data into predictions.
        Returns:
            pred: A tensor of shape (batch_size, n_classes)
        """
        activation = tf.nn.tanh
        xavier = tf.contrib.layers.xavier_initializer()
        config = self.config

        conv1 = tf.layers.conv2d(inputs=self.perm_placeholder, filters=config.n_filters,kernel_size=[config.kernel_size,config.kernel_size],kernel_initializer=xavier, padding="same",activation=activation)
        normed_conv1 = tf.layers.batch_normalization(conv1,axis = -1,center = True, scale = True, training = self.is_training,trainable = True)
        dropout_conv1 = tf.layers.dropout(inputs = normed_conv1,rate=config.dropout,training = self.is_training)

        conv1_pool1 = tf.layers.max_pooling2d(inputs=dropout_conv1, pool_size=[2,2], strides=2)
        conv2_pool1 = tf.layers.conv2d(inputs = conv1_pool1, filters=config.n_filters,kernel_size=[config.kernel_size,config.kernel_size],kernel_initializer=xavier, padding="same",activation=activation)
        normed_conv2_pool1 = tf.layers.batch_normalization(conv2_pool1,axis = -1,center = True, scale = True, training = self.is_training,trainable = True)
        conv2_pool1_upscaled = tf.image.resize_images(normed_conv2_pool1, [config.nx, config.nx])  

        conv1_pool2 = tf.layers.max_pooling2d(inputs=conv1_pool1, pool_size=[2,2], strides=2)
        conv2_pool2 = tf.layers.conv2d(inputs = conv1_pool2, filters=config.n_filters,kernel_size=[config.kernel_size,config.kernel_size], kernel_initializer=xavier,padding="same",activation=activation)
        normed_conv2_pool2 = tf.layers.batch_normalization(conv2_pool2,axis = -1,center = True, scale = True, training = self.is_training,trainable = True)
        conv2_pool2_upscaled = tf.image.resize_images(normed_conv2_pool2, [config.nx, config.nx])   

        conv1_pool3 = tf.layers.max_pooling2d(inputs=conv1_pool2, pool_size=[2,2], strides=2)
        conv2_pool3 = tf.layers.conv2d(inputs = conv1_pool3, filters=config.n_filters,kernel_size=[config.kernel_size,config.kernel_size], kernel_initializer=xavier,padding="same",activation=activation)
        normed_conv2_pool3 = tf.layers.batch_normalization(conv2_pool3,axis = -1,center = True, scale = True, training = self.is_training,trainable = True)
        conv2_pool3_upscaled = tf.image.resize_images(normed_conv2_pool3, [config.nx, config.nx])   

        conv2 = tf.layers.conv2d(inputs=dropout_conv1, filters=config.n_filters,kernel_size=[config.kernel_size,config.kernel_size], kernel_initializer=xavier,padding="same",activation=activation)
        conv3 = tf.layers.conv2d(inputs=conv2, filters=config.n_filters,kernel_size=[config.kernel_size,config.kernel_size],kernel_initializer=xavier, padding="same",activation=activation)

        weights = tf.get_variable("weights", shape=[4,],initializer=tf.contrib.layers.xavier_initializer())

        conv_comb = weights[0]*conv3 + weights[1]*conv2_pool1_upscaled+ weights[2]*conv2_pool2_upscaled + weights[3]*conv2_pool3_upscaled
        conv4 = tf.layers.conv2d(inputs=conv_comb, filters=config.n_filters,kernel_size=[1,1],kernel_initializer=xavier, padding="same",activation=activation)
        pres = tf.layers.conv2d(inputs=conv4, filters=1,kernel_size=[1,1],kernel_initializer=xavier, padding="same",activation = activation)

        pres_flat = tf.reshape(pres,[-1,config.nx*config.nx,1])*config.max_val + config.mean_val
        #pred = tf.sparse_tensor_dense_matmul(self.U_face_operator_placeholder,pres_flat) + self.U_face_fixed_placeholder
        v_pred = tf.matmul(tf.sparse_tensor_to_dense(tf.sparse_reorder(self.U_face_operator_placeholder)),pres_flat) + tf.reshape(self.U_face_fixed_placeholder,[-1,config.nfaces,1])
        v_pred = tf.reshape(v_pred,[-1,config.nfaces])
        return v_pred, pres

    def test_matrix_op(self, sess, input_set):
        # pass the dev set and calculate the face velocity from the input pressure and operator
        input_p_flat = tf.reshape(self.pressure_placeholder, [-1,config.nx*config.nx,1])
        u_test = tf.matmul(tf.sparse_tensor_to_dense(tf.sparse_reorder(self.U_face_operator_placeholder)),input_p_flat)
        u_test_reshaped = tf.reshape(u_test,[-1,config.nfaces])
        u_actual_reshaped = tf.reshape(self.U_face_placeholder, [-1, self.config.nfaces])
        u_diff = u_test_reshaped - u_actual_reshaped
        perm_dev, U_face_fixed_dev, U_face_operator_dev, U_pressure_dev, U_face_dev =  zip(*input_set)
        feed = self.create_feed_dict(perm_dev, U_face_fixed_dev, U_face_operator_dev, True, U_pressure_dev, U_face_dev)
        diff = sess.run(u_diff, feed_dict=feed)
        return diff

    def add_loss_op(self, pred_velocity,pred_pres):
        """Adds Ops for the loss function to the computational graph.
        Args:
            pred: A tensor of shape (batch_size, n_classes)
        Returns:
            loss: A 0-d tensor (scalar) output
        """
        v_weight = config.weight
        pred_velocity= tf.reshape(pred_velocity,[-1,self.config.nfaces])
        actual = tf.reshape(self.U_face_placeholder, [-1, self.config.nfaces])
        loss = (1.0 - v_weight) * tf.nn.l2_loss(pred_pres-self.pressure_placeholder) + \
               v_weight * tf.nn.l2_loss(pred_velocity-actual) #
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

    def train_on_batch(self, sess, perm_batch,U_face_fixed_batch,U_face_operator_batch,U_pressure_batch,U_face_batch):
        """Perform one step of gradient descent on the provided batch of data.
        Args:
            sess: tf.Session()
            input_batch: np.ndarray of shape (n_samples, n_features)
            labels_batch: np.ndarray of shape (n_samples, n_classes)
        Returns:
            loss: loss over the batch (a scalar)
        """
        feed = self.create_feed_dict(perm_batch,U_face_fixed_batch,U_face_operator_batch, True,U_pressure_batch,U_face_batch)
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        return loss/(len(perm_batch))

    def predict_on_batch(self, sess, perm_batch,U_face_fixed_batch,U_face_operator_batch):
        """Make predictions for the provided batch of data
        Args:
            sess: tf.Session()
            input_batch: np.ndarray of shape (n_samples, n_features)
        Returns:
            predictions: np.ndarray of shape (n_samples, n_classes)
        """
        feed = self.create_feed_dict(perm_batch,U_face_fixed_batch,U_face_operator_batch,False)
        predictions, pres = sess.run([self.pred, self.pres], feed_dict=feed)
        # pres = sess.run(self.pres, feed_dict=feed)
        # u_diff = sess.run()
        return predictions, pres

    def fit(self,sess,train_set,dev_set):
        save_dir = self.config.model_save_dir
        save_file = os.path.join(save_dir, 'best_validation_model')
        best_dev = 1e9
        best_velocity = None
        best_pres = None

        for epoch in range(self.config.n_epochs):
            self.epoch_count = epoch
            print ("Epoch {:} out of {:}".format(epoch + 1, self.config.n_epochs))
            dev_score, pred , pres= self.run_epoch(sess,train_set,dev_set)

            if dev_score < best_dev:
                best_dev = dev_score
                best_velocity = pred
                best_pres = pres
                perm_dev, U_face_fixed_dev, U_face_operator_dev, U_pressure_dev, U_face_dev = zip(*dev_set)
                np.savez(save_file,best_velocity=best_velocity,best_pres = best_pres,perm=perm_dev,U_face_fixed = U_face_fixed_dev,U_face_operator = U_face_operator_dev,
                         pressure=U_pressure_dev,U_face= U_face_dev)
                print("new best norm found {:}".format(best_dev))

    def run_epoch(self,sess,train_examples,dev_set):
        save_dir = self.config.model_save_dir
        init_file = os.path.join(save_dir, 'first_train_model')
        latest_file = os.path.join(save_dir, 'latest_train_model')
        config = self.config
        num_train = len(train_examples)
        ind = np.arange(num_train)
        np.random.shuffle(ind)
        ind.astype(int)
        batchNum = 1


        perm_dev, U_face_fixed_dev, U_face_operator_dev, U_pressure_dev, U_face_dev =  zip(*dev_set)
        pred, pres = self.predict_on_batch(sess,perm_dev, U_face_fixed_dev, U_face_operator_dev)
        np.savez(init_file,best_velocity=pred,best_pres =pres, perm=perm_dev,U_face_fixed = U_face_fixed_dev,U_face_operator = U_face_operator_dev,
                         pressure=U_pressure_dev,U_face= U_face_dev)

        num_batch = int(np.ceil(num_train/config.batch_size))
        for i in range(0, num_train,config.batch_size):
            batch_train = [ train_examples[i] for i in ind[i:i+config.batch_size]]
            perm_train, U_face_fixed_train, U_face_operator_train, U_pressure_train, U_face_train =  zip(*batch_train)

            loss = self.train_on_batch(sess,perm_train, U_face_fixed_train, U_face_operator_train, U_pressure_train, U_face_train)
            self.recordOutput(loss,batchNum)

            print("Loss for Batch {:} out of {:} is: {:}".format(batchNum,num_batch,loss))
            batchNum += 1

        pred_train ,pres_train= self.predict_on_batch(sess,perm_train, U_face_fixed_train, U_face_operator_train)
        np.savez(latest_file,pred_train=pred_train,pres_train=pres_train,perm=perm_train,U_face_fixed = U_face_fixed_train,U_face_operator = U_face_operator_train,
                 pressure=U_pressure_train,U_face = U_face_train)

        perm_dev, U_face_fixed_dev, U_face_operator_dev, U_pressure_dev, U_face_dev =  zip(*dev_set)
        pred, pres = self.predict_on_batch(sess,perm_dev, U_face_fixed_dev, U_face_operator_dev)
        norms = []
        for i in range(len(U_face_dev)):
            norms.append(np.sum(np.power(U_face_dev[i]-pred[i,:],2)))
        return np.sum(norms)*0.5/(len(U_face_dev)), pred, pres

    def recordOutput(self,loss,batchNum):
        self.loss_history.append(loss)
        self.iter_number.append(float(self.epoch_count)+np.float(batchNum*self.config.batch_size/np.float(self.config.n_train)))

    def save_loss_history(self, save_folder):
        # np.savez(os.path.join(save_folder, 'loss'), loss=self.loss_history, iter_number=self.iter_number)
        fig, ax = plt.subplots(1,1)
        ax.plot(self.iter_number, self.loss_history)
        # ax.plot(self.loss_history)
        fig.savefig(os.path.join(save_folder, 'loss.png'), format='png')

    def build(self):
        self.add_placeholders()
        self.pred, self.pres = self.add_prediction_op()
        self.loss = self.add_loss_op(self.pred,self.pres)
        self.train_op = self.add_training_op(self.loss)
        self.loss_history = []
        self.iter_number = []

    def __init__(self, config):
        self.config = config
        self.build()
    def saveLossHistory(self):
        np.savez("velocity_loss_history_both",loss_history=self.loss_history,epoch_num = self.iter_number)

from os.path import dirname
class Config():
    weight = 0.5
    lr = 1e-3
    n_epochs = 100
    kernel_size = 9
    batch_size = 30
    n_filters = 16
    dropout = 0.2
    proj_folder = dirname(dirname(dirname(os.path.realpath(__file__))))
    model_save_dir = os.path.join(proj_folder, 'temp', 'both_overfit', 'models')

if __name__ == "__main__":
    print("Started running")
    config = Config()
    proj_folder = dirname(dirname(dirname(os.path.realpath(__file__))))
    print(proj_folder)
    datFile = np.load(os.path.join(proj_folder,'data','data_64_nonperiodic.npz'))
    X = datFile['X']
    Y_in = datFile['Y']
    # TODO: normalization removed
    # U_face = datFile['U_face']/0.1
    U_face = datFile['U_face']
    U_face_operator = datFile['U_face_operator']
    U_face_fixed = datFile['U_face_fixed']
    config.nx = X.shape[1]
    config.nfaces = U_face_fixed.shape[1]

    config.mean_val = np.mean(Y_in)
    config.max_val = np.max(np.fabs(Y_in-config.mean_val))*(4/3)
    # TODO: normalization removed
    # Y = (Y_in-config.mean_val)/config.max_val
    Y = Y_in
    n_train = 2
    n_dev = 1#X.shape[0]-n_train

    train_set = []
    for i in range(n_train):
        train_set.append((X[i, :, :, :], U_face_fixed[i, :], U_face_operator[i], Y[i, :, :, :], U_face[i, :]))

    dev_set = []
    for i in range(n_dev):
        dev_set.append((X[i+n_train, :, :, :], U_face_fixed[i+n_train, :], U_face_operator[i+n_train], Y[i+n_train, :, :, :], U_face[i+n_train, :]))

    config.n_train = n_train

    with tf.Graph().as_default():
        print ("building Model")
        model = NNModel(config)
        init = tf.global_variables_initializer()
        with tf.Session() as session:
            session.run(init)
            diff = model.test_matrix_op(session, train_set)
            print('error for reproducing face velocity: ', np.linalg.norm(diff))
            model.fit(session, train_set, dev_set)
            model.save_loss_history(os.path.join(proj_folder, 'temp', 'both_overfit', 'pics'))


    #################################################################################################
    # plot the results
    import numpy as np
    import os as os
    import pickle as pickle
    from deepres.plotting.side_by_side import side_by_side
    from os.path import dirname

    print('comparing the pressure solutions...')
    proj_folder = dirname(dirname(dirname(os.path.realpath(__file__))))
    # path to save the training data file
    save_folder = os.path.join(proj_folder, 'temp', 'both_overfit', 'pics')
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    # grid_path = os.path.join(proj_folder, 'sample_scripts', 'script_files', '100_100_periodic.pkl')
    grid_path = os.path.join(proj_folder, 'data', 'grids', '64_64_periodic.pkl')
    model_folder = os.path.join(proj_folder, 'temp', 'both_overfit', 'models')
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
