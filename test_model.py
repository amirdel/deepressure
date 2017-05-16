import tensorflow as tf
import numpy as np
import random
from model import Model

class Config():
    nx = 100
    lr = 0.0001
    n_epochs =10
    kernel_size = 3
    batch_size = 2
    n_filters = 16
    dropout = 0.2
    nfaces = 100

class TestModel(Model):
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
            indices_batch = np.concatenate((batch_num, row,col),axis=1)
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
        config = self.config

        conv1 = tf.layers.conv2d(inputs=self.perm_placeholder, filters=config.n_filters,kernel_size=[config.kernel_size,config.kernel_size],kernel_initializer=tf.contrib.layers.xavier_initializer(), padding="same",activation=tf.nn.relu)
        normed_conv1 = tf.layers.batch_normalization(conv1,axis = -1,center = True, scale = True, training = self.is_training,trainable = True)
        dropout_conv1 = tf.layers.dropout(inputs = normed_conv1,rate=config.dropout,training = self.is_training)

        conv1_pool1 = tf.layers.max_pooling2d(inputs=dropout_conv1, pool_size=[2,2], strides=2)
        conv2_pool1 = tf.layers.conv2d(inputs = conv1_pool1, filters=config.n_filters,kernel_size=[config.kernel_size,config.kernel_size],kernel_initializer=tf.contrib.layers.xavier_initializer(), padding="same",activation=tf.nn.relu)
        conv2_pool1_upscaled = tf.image.resize_images(conv2_pool1, [config.nx, config.nx])  

        conv1_pool2 = tf.layers.max_pooling2d(inputs=conv1_pool1, pool_size=[2,2], strides=2)
        conv2_pool2 = tf.layers.conv2d(inputs = conv1_pool2, filters=config.n_filters,kernel_size=[config.kernel_size,config.kernel_size], kernel_initializer=tf.contrib.layers.xavier_initializer(),padding="same",activation=tf.nn.relu)
        conv2_pool2_upscaled = tf.image.resize_images(conv2_pool2, [config.nx, config.nx])   

        conv1_pool3 = tf.layers.max_pooling2d(inputs=conv1_pool2, pool_size=[2,2], strides=2)
        conv2_pool3 = tf.layers.conv2d(inputs = conv1_pool3, filters=config.n_filters,kernel_size=[config.kernel_size,config.kernel_size], kernel_initializer=tf.contrib.layers.xavier_initializer(),padding="same",activation=tf.nn.relu)
        conv2_pool3_upscaled = tf.image.resize_images(conv2_pool3, [config.nx, config.nx])   

        conv2 = tf.layers.conv2d(inputs=dropout_conv1, filters=config.n_filters,kernel_size=[config.kernel_size,config.kernel_size], kernel_initializer=tf.contrib.layers.xavier_initializer(),padding="same",activation=tf.nn.relu)
        conv3 = tf.layers.conv2d(inputs=conv2, filters=config.n_filters,kernel_size=[config.kernel_size,config.kernel_size],kernel_initializer=tf.contrib.layers.xavier_initializer(), padding="same",activation=tf.nn.relu)

        weights = tf.get_variable("weights", shape=[4,],initializer=tf.contrib.layers.xavier_initializer())

        conv_comb = weights[0]*conv3 + weights[1]*conv2_pool1_upscaled+ weights[2]*conv2_pool2_upscaled + weights[3]*conv2_pool3_upscaled
        conv4 = tf.layers.conv2d(inputs=conv_comb, filters=config.n_filters,kernel_size=[1,1],kernel_initializer=tf.contrib.layers.xavier_initializer(), padding="same",activation=tf.nn.relu)
        pres = tf.layers.conv2d(inputs=conv4, filters=1,kernel_size=[1,1],kernel_initializer=tf.contrib.layers.xavier_initializer(), padding="same",activation = tf.nn.tanh)

        pres_flat = tf.reshape(pres,[-1,config.nx*config.nx,1])
        #pred = tf.sparse_tensor_dense_matmul(self.U_face_operator_placeholder,pres_flat) + self.U_face_fixed_placeholder
        pred = tf.matmul(tf.sparse_tensor_to_dense(tf.sparse_reorder(self.U_face_operator_placeholder)),pres_flat) + tf.reshape(self.U_face_fixed_placeholder,[-1,config.nfaces,1])
        #pred = pres_flat
        pred = tf.reshape(pred,[-1,config.nfaces])
        return pred, pres


    def add_loss_op(self, pred):
        """Adds Ops for the loss function to the computational graph.
        Args:
            pred: A tensor of shape (batch_size, n_classes)
        Returns:
            loss: A 0-d tensor (scalar) output
        """
        print(pred)
        predictions= tf.reshape(pred,[-1,self.config.nfaces])

        actual = tf.reshape(self.U_face_placeholder, [-1, self.config.nfaces])

        loss = tf.nn.l2_loss(predictions-actual)
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
        predictions = sess.run(self.pred, feed_dict=feed)
        pres = sess.run(self.pres, feed_dict=feed)
        return predictions, pres

    def fit(self,sess,train_set,dev_set):
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
                np.savez("best_velocity",best_velocity=best_velocity,best_pres = best_pres,perm=perm_dev,U_face_fixed = U_face_fixed_dev,U_face_operator = U_face_operator_dev,
                         pressure=U_pressure_dev,U_face= U_face_dev)
                print("new best norm found {:}".format(best_dev))

    def run_epoch(self,sess,train_examples,dev_set):
        config = self.config
        num_train = len(train_examples)
        ind = np.arange(num_train)
        np.random.shuffle(ind)
        ind.astype(int)
        batchNum = 1


        perm_dev, U_face_fixed_dev, U_face_operator_dev, U_pressure_dev, U_face_dev =  zip(*dev_set)
        pred, pres = self.predict_on_batch(sess,perm_dev, U_face_fixed_dev, U_face_operator_dev)
        np.savez("inital_pred",best_velocity=pred,best_pres =pres, perm=perm_dev,U_face_fixed = U_face_fixed_dev,U_face_operator = U_face_operator_dev,
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
        np.savez("train_pred",pred_train=pred_train,pres_train=pres_train,perm=perm_train,U_face_fixed = U_face_fixed_train,U_face_operator = U_face_operator_train,
                 pressure=U_pressure_train,U_face = U_face_train)

        perm_dev, U_face_fixed_dev, U_face_operator_dev, U_pressure_dev, U_face_dev =  zip(*dev_set)
        pred, pres = self.predict_on_batch(sess,perm_dev, U_face_fixed_dev, U_face_operator_dev)
        norms = []
        for i in range(len(U_face_dev)):
            norms.append(np.sum(np.power(U_face_dev[i]-pred[i,:],2)))
        return np.sum(norms)*0.5/(len(U_face_dev)), pred, pres

    def recordOutput(self,loss,batchNum):
        self.loss_history = np.append(self.loss_history,[loss])
        self.epoch_num = np.append(self.epoch_num,[np.float(self.epoch_count)+np.float(batchNum*self.config.batch_size/np.float(self.config.n_train))])

    def build(self):
        self.add_placeholders()
        self.pred, self.pres = self.add_prediction_op()
        self.loss = self.add_loss_op(self.pred)
        self.train_op = self.add_training_op(self.loss)
        self.loss_history = np.array([])
        self.epoch_num = np.array([])
    def __init__(self, config):
        self.config = config
        self.build()
    def saveLossHistory(self):
        np.savez("velocity_loss_history",loss_history=self.loss_history,epoch_num = self.epoch_num)


if __name__ == "__main__":
    print("Started running")
    config = Config()

    datFile = np.load('temp/data.npz')
    X = datFile['X']
    Y = datFile['Y']
    U_face = datFile['U_face']/0.1
    U_face_operator = datFile['U_face_operator']
    U_face_fixed = datFile['U_face_fixed']
    config.nx = X.shape[1]
    config.nfaces = U_face_fixed.shape[1]

    n_train = 495# int(X.shape[0])-n_dev
    n_dev = 5#X.shape[0]-n_train

    train_set = []
    for i in range(n_train):
        train_set.append((X[i, :, :, :], U_face_fixed[i, :], U_face_operator[i], Y[i, :, :, :], U_face[i, :]))

    dev_set = []
    for i in range(n_dev):
        dev_set.append((X[i+n_train, :, :, :], U_face_fixed[i+n_train, :], U_face_operator[i+n_train], Y[i+n_train, :, :, :], U_face[i+n_train, :]))

    config.n_train = n_train

    with tf.Graph().as_default():
        print ("building Model")
        model = TestModel(config)
        init = tf.global_variables_initializer()
        with tf.Session() as session:
            session.run(init)
            model.fit(session, train_set, dev_set)
            model.saveLossHistory()
