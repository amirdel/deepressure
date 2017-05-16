import tensorflow as tf
import numpy as np
import random
from model import Model

class Config():
    nx = 100
    lr = 0.0001
    kernel_size= 9
    n_epochs =10
    batch_size = 30
    n_filters = 16
    dropout = 0.2

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

        self.input_placeholder   = tf.placeholder(tf.float32,   (None, self.config.nx,self.config.nx,1), name = "input")
        self.labels_placeholder  = tf.placeholder(tf.float32,   (None, self.config.nx,self.config.nx,1), name = "out")
        self.is_training = tf.placeholder(tf.bool)
    def create_feed_dict(self, inputs_batch, is_training,labels_batch=None):
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
        if labels_batch is None:
            feed_dict = {
                self.input_placeholder: inputs_batch,
                self.is_training: is_training,
            }
        else:
            feed_dict = {
                self.input_placeholder: inputs_batch,
                self.labels_placeholder: labels_batch,
                self.is_training: is_training
            }
        ### END YOUR CODE
        return feed_dict

    def add_prediction_op(self):
        """Implements the core of the model that transforms a batch of input data into predictions.
        Returns:
            pred: A tensor of shape (batch_size, n_classes)
        """
        config = self.config

        conv1 = tf.layers.conv2d(inputs=self.input_placeholder, filters=config.n_filters,kernel_size=[config.kernel_size,config.kernel_size],kernel_initializer=tf.contrib.layers.xavier_initializer(), padding="same",activation=tf.nn.tanh)
        normed_conv1 = tf.layers.batch_normalization(conv1,axis = -1,center = True, scale = True, training = self.is_training,trainable = True)
        dropout_conv1 = tf.layers.dropout(inputs = normed_conv1,rate=config.dropout,training = self.is_training)

        conv1_pool1 = tf.layers.max_pooling2d(inputs=dropout_conv1, pool_size=[2,2], strides=2)
        conv2_pool1 = tf.layers.conv2d(inputs = conv1_pool1, filters=config.n_filters,kernel_size=[config.kernel_size,config.kernel_size],kernel_initializer=tf.contrib.layers.xavier_initializer(), padding="same",activation=tf.nn.tanh)
        conv2_pool1_upscaled = tf.image.resize_images(conv2_pool1, [config.nx, config.nx])  

        conv1_pool2 = tf.layers.max_pooling2d(inputs=conv1_pool1, pool_size=[2,2], strides=2)
        conv2_pool2 = tf.layers.conv2d(inputs = conv1_pool2, filters=config.n_filters,kernel_size=[config.kernel_size,config.kernel_size], kernel_initializer=tf.contrib.layers.xavier_initializer(),padding="same",activation=tf.nn.tanh)
        conv2_pool2_upscaled = tf.image.resize_images(conv2_pool2, [config.nx, config.nx])   

        conv1_pool3 = tf.layers.max_pooling2d(inputs=conv1_pool2, pool_size=[2,2], strides=2)
        conv2_pool3 = tf.layers.conv2d(inputs = conv1_pool3, filters=config.n_filters,kernel_size=[config.kernel_size,config.kernel_size], kernel_initializer=tf.contrib.layers.xavier_initializer(),padding="same",activation=tf.nn.tanh)
        conv2_pool3_upscaled = tf.image.resize_images(conv2_pool3, [config.nx, config.nx])   

        conv2 = tf.layers.conv2d(inputs=dropout_conv1, filters=config.n_filters,kernel_size=[config.kernel_size,config.kernel_size], kernel_initializer=tf.contrib.layers.xavier_initializer(),padding="same",activation=tf.nn.tanh)
        conv3 = tf.layers.conv2d(inputs=conv2, filters=config.n_filters,kernel_size=[config.kernel_size,config.kernel_size],kernel_initializer=tf.contrib.layers.xavier_initializer(), padding="same",activation=tf.nn.tanh)

        weights = tf.get_variable("weights", shape=[4,],initializer=tf.contrib.layers.xavier_initializer())

        conv_comb = weights[0]*conv3 + weights[1]*conv2_pool1_upscaled+ weights[2]*conv2_pool2_upscaled + weights[3]*conv2_pool3_upscaled
        conv4 = tf.layers.conv2d(inputs=conv_comb, filters=config.n_filters,kernel_size=[1,1],kernel_initializer=tf.contrib.layers.xavier_initializer(), padding="same",activation=tf.nn.tanh)
        pred = tf.layers.conv2d(inputs=conv4, filters=1,kernel_size=[1,1],kernel_initializer=tf.contrib.layers.xavier_initializer(), padding="same")
        
        return pred


    def add_loss_op(self, pred):
        """Adds Ops for the loss function to the computational graph.
        Args:
            pred: A tensor of shape (batch_size, n_classes)
        Returns:
            loss: A 0-d tensor (scalar) output
        """
        loss = tf.nn.l2_loss(pred-self.labels_placeholder)
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

    def train_on_batch(self, sess, inputs_batch, labels_batch):
        """Perform one step of gradient descent on the provided batch of data.
        Args:
            sess: tf.Session()
            input_batch: np.ndarray of shape (n_samples, n_features)
            labels_batch: np.ndarray of shape (n_samples, n_classes)
        Returns:
            loss: loss over the batch (a scalar)
        """
        feed = self.create_feed_dict(inputs_batch, True,labels_batch=labels_batch)
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        return loss/(len(inputs_batch))

    def predict_on_batch(self, sess, inputs_batch):
        """Make predictions for the provided batch of data
        Args:
            sess: tf.Session()
            input_batch: np.ndarray of shape (n_samples, n_features)
        Returns:
            predictions: np.ndarray of shape (n_samples, n_classes)
        """
        feed = self.create_feed_dict(inputs_batch,False)
        predictions = sess.run(self.pred, feed_dict=feed)
        return predictions

    def fit(self,sess,train_examples,dev_set):
        best_dev = 1e9
        best_pred = None
        for epoch in range(self.config.n_epochs):
            print ("Epoch {:} out of {:}".format(epoch + 1, self.config.n_epochs))
            dev_score, pred = self.run_epoch(sess,train_examples,dev_set)
            if dev_score < best_dev:
                best_dev = dev_score
                best_pred = pred
                input_dev, labels_dev = zip(*dev_set)
                np.savez("best_pred_pressure",best_pred=best_pred,perm=input_dev,pressure=labels_dev)
                print("new best norm found {:}".format(best_dev))

    def run_epoch(self,sess,train_examples,dev_set):
        config = self.config
        num_train = len(train_examples)
        ind = np.arange(num_train)
        np.random.shuffle(ind)
        ind.astype(int)
        batchNum = 1
        num_batch = int(np.ceil(num_train/config.batch_size))
        for i in range(0, num_train,config.batch_size):
            batch_train = [ train_examples[i] for i in ind[i:i+config.batch_size]]
            input_train , labels_train =  zip(*batch_train)
            loss = self.train_on_batch(sess,input_train,labels_train)
            print("Loss for Batch {:} out of {:} is: {:}".format(batchNum,num_batch,loss))
            batchNum += 1

        input_dev , labels_dev =  zip(*dev_set)
        pred = self.predict_on_batch(sess,input_dev)
        norms = []
        for i in range(len(labels_dev)):
            norms.append(np.sum(np.power(labels_dev[i]-pred[i,:,:,:],2)))
        return np.sum(norms)*0.5/len(labels_dev), pred


    def build(self):
        self.add_placeholders()
        self.pred = self.add_prediction_op()
        self.loss = self.add_loss_op(self.pred)
        self.train_op = self.add_training_op(self.loss)
    def __init__(self, config):
        self.config = config
        self.build()

if __name__ == "__main__":
    print("Started running")
    config = Config()

    datFile = np.load('temp/data.npz')
    X = datFile['X']
    Y = datFile['Y']/0.05

    n_train = int(X.shape[0]*0.8)
    n_dev = X.shape[0]-n_train
    train_perm = X[:n_train]
    train_pres = Y[:n_train] 
    train_examples = []
    for i in range(n_train):
        train_examples.append((train_perm[i,:,:,:],train_pres[i,:,:,:]))
    n_dev = 5
    dev_perm = X[n_train+1:]
    dev_pres = Y[n_train+1:]    
    dev_set = []
    for i in range(n_dev):
        dev_set.append((dev_perm[i,:,:,:],dev_pres[i,:,:,:]))


    with tf.Graph().as_default():
        print ("building Model")
        model = TestModel(config)
        init = tf.global_variables_initializer()
        with tf.Session() as session:
            session.run(init)
            model.fit(session, train_examples, dev_set)