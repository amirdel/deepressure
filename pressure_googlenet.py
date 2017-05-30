import tensorflow as tf
import numpy as np
import random
from model import Model

class Config():
    nx = 64
    lr = 0.0001
    kernel_size= 3
    n_epochs =5000
    batch_size = 30
    n_filters = 64
    dropout = 0.1

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

        conv1_7x7_s2 = tf.layers.conv2d(self.input_placeholder, filters=64,kernel_size=[7,7],kernel_initializer=tf.contrib.layers.xavier_initializer(), padding="same",activation=tf.nn.relu)
        pool1_3x3_s2 = tf.layers.max_pooling2d(inputs=conv1_7x7_s2, pool_size=[3,3], strides=2, padding = 'same')
        pool1_norm1 = tf.nn.lrn(pool1_3x3_s2)
        conv2_3x3_reduce = tf.layers.conv2d(pool1_norm1, filters=64,kernel_size=[1,1],kernel_initializer=tf.contrib.layers.xavier_initializer(), padding="same",activation=tf.nn.relu)
        conv2_3x3 = tf.layers.conv2d(conv2_3x3_reduce ,filters=192,kernel_size=[3,3],kernel_initializer=tf.contrib.layers.xavier_initializer(), padding="same",activation=tf.nn.relu)
        conv2_norm2 = tf.nn.lrn(conv2_3x3)
        pool2_3x3_s2 = tf.layers.max_pooling2d(inputs=conv2_norm2, pool_size=[3,3], strides=2, padding = 'same')
        
        conv3a = tf.layers.conv2d(conv2_3x3_reduce, filters=128,kernel_size=[1,1],kernel_initializer=tf.contrib.layers.xavier_initializer(), padding="same",activation=tf.nn.relu)
        conv3a_upscaled = tf.image.resize_images(conv3a, [config.nx, config.nx])         

        conv3b = tf.layers.conv2d(conv2_3x3_reduce ,filters=128,kernel_size=[1,1],kernel_initializer=tf.contrib.layers.xavier_initializer(), padding="same",activation=tf.nn.relu)
        conv4b = tf.layers.conv2d(conv3b ,filters=196,kernel_size=[3,3],kernel_initializer=tf.contrib.layers.xavier_initializer(), padding="same",activation=tf.nn.relu)
        conv4b_upscaled = tf.image.resize_images(conv4b, [config.nx, config.nx])         

        conv3c = tf.layers.conv2d(conv2_3x3_reduce, filters=128,kernel_size=[1,1],kernel_initializer=tf.contrib.layers.xavier_initializer(), padding="same",activation=tf.nn.relu)
        conv4c = tf.layers.conv2d(conv3c ,filters=128,kernel_size=[5,5],kernel_initializer=tf.contrib.layers.xavier_initializer(), padding="same",activation=tf.nn.relu)
        conv4c_upscaled = tf.image.resize_images(conv4c, [config.nx, config.nx])          
        pool3 = tf.layers.max_pooling2d(inputs=pool2_3x3_s2, pool_size=[3,3], strides=1, padding = 'same')
        pool3_conv1 = tf.layers.conv2d(pool3, filters=128,kernel_size=[1,1],kernel_initializer=tf.contrib.layers.xavier_initializer(), padding="same",activation=tf.nn.relu)
        conv2_pool3_upscaled = tf.image.resize_images(pool3_conv1, [config.nx, config.nx])   

        inception1 = tf.nn.relu(tf.concat([conv1_7x7_s2,conv3a_upscaled,conv4b_upscaled,conv4c_upscaled,conv2_pool3_upscaled], axis=3))
        
        inception1_conv = tf.layers.conv2d(inception1 ,filters=128,kernel_size=[3,3],kernel_initializer=tf.contrib.layers.xavier_initializer(), padding="same",activation=tf.nn.relu)
        inception1_conv = tf.layers.conv2d(inception1 ,filters=192,kernel_size=[1,1],kernel_initializer=tf.contrib.layers.xavier_initializer(), padding="same",activation=tf.nn.relu)
        
        pred = tf.layers.conv2d(inputs=inception1, filters=1,kernel_size=[1,1],kernel_initializer=tf.contrib.layers.xavier_initializer(), padding="same")
        
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
                np.savez("best_pred_google",best_pred=best_pred,perm=input_dev,pressure=labels_dev)
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

        pred_train = self.predict_on_batch(sess,input_train)
        np.savez("train_google",best_pred=pred_train,perm=input_train,pressure=labels_train)

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

    datFile = np.load('temp/data_64_nonperiodic.npz')
    X = datFile['X']
    Y_in = datFile['Y']
    max_val = np.max(np.fabs(Y_in))*0.75
    print(max_val)
    mean_val = np.mean(Y_in)
    print(mean_val)
    Y = (Y_in-mean_val)/max_val
    n_train = 7#int(X.shape[0]*0.8)

    train_perm = X[:n_train]
    train_pres = Y[:n_train] 
    train_examples = []
    for i in range(n_train):
        train_examples.append((train_perm[i,:,:,:],train_pres[i,:,:,:]))
    n_dev = 2 #X.shape[0]-n_train
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