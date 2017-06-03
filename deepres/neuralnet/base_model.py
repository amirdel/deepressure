import tensorflow as tf
import numpy as np
from model import Model
import os as os
import matplotlib.pyplot as plt

class BaseModel(Model):
    def add_placeholders(self):
        """Adds placeholder variables to tensorflow computational graph.
        Tensorflow uses placeholder variables to represent locations in a
        computational graph where data is inserted.  These placeholders are used as
        inputs by the rest of the model building and will be fed data during
        training.
        See for more information:
        https://www.tensorflow.org/versions/r0.7/api_docs/python/io_ops.html#placeholders
        """

        self.perm_placeholder = tf.placeholder(tf.float32, (None, self.config.nx, self.config.nx, 1), name="perm")
        self.pressure_placeholder = tf.placeholder(tf.float32, (None, self.config.nx, self.config.nx, 1),
                                                   name="pressure")
        self.Div_U_operator_placeholder = tf.sparse_placeholder(tf.float32, (
        None, self.config.nfaces, self.config.nx * self.config.nx), name="U_face_operator")
        self.is_training = tf.placeholder(tf.bool)

    def create_feed_dict(self, perm_batch, Div_U_operator_batch, is_training, U_pressure_batch=None):
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
            batch_num = i * np.ones((len(Div_U_operator_batch[i].col), 1))
            row = Div_U_operator_batch[i].row.reshape(len(batch_num), 1)
            col = Div_U_operator_batch[i].col.reshape(len(batch_num), 1)
            indices_batch = np.concatenate((batch_num, row, col), axis=1)
            if i == 0:
                indices = indices_batch
                values = Div_U_operator_batch[i].data
            else:
                indices = np.concatenate((indices, indices_batch))
                values = np.concatenate((values, Div_U_operator_batch[i].data))

        shape = np.array([n_batch, config.nfaces, config.nx * config.nx], dtype=np.int64)
        indices = indices.astype(int)
        if U_pressure_batch is not None:
            feed_dict = {
                self.perm_placeholder: perm_batch,
                self.Div_U_operator_placeholder: (indices, values, shape),
                self.pressure_placeholder: U_pressure_batch,
                self.is_training: is_training,
            }
        else:
            feed_dict = {
                self.perm_placeholder: perm_batch,
                self.Div_U_operator_placeholder: (indices, values, shape),
                self.is_training: is_training,
            }
        return feed_dict

    def test_matrix_op(self, sess, input_set):
        config = self.config
        # pass the dev set and calculate the face velocity from the input pressure and operator
        input_p_flat = tf.reshape(self.pressure_placeholder, [-1, config.nx * config.nx, 1])
        div_u = tf.matmul(tf.sparse_tensor_to_dense(tf.sparse_reorder(self.Div_U_operator_placeholder)), input_p_flat)
        div_u_reshaped = tf.reshape(div_u, [-1, config.nfaces])
        perm_dev, Div_U_operator_dev, U_pressure_dev = zip(*input_set)
        feed = self.create_feed_dict(perm_dev, Div_U_operator_dev, True, U_pressure_dev)
        diff = sess.run(div_u_reshaped, feed_dict=feed)
        return diff

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
        feed = self.create_feed_dict(perm_batch, Div_U_operator_batch, True, U_pressure_batch)
        # _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        _, loss, loss_ratio = sess.run([self.train_op, self.loss, self.loss_ratio], feed_dict=feed)
        return loss / (len(perm_batch)), loss_ratio

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

    def fit(self, sess, train_set, dev_set, reduce_every=2):
        config = self.config
        save_dir = self.config.model_save_dir
        save_file = os.path.join(save_dir, 'best_validation_model')
        best_dev = 1e9
        best_velocity = None
        best_pres = None

        for epoch in range(config.n_epochs):
            if epoch > 0 and (not epoch % reduce_every):
                config.lr = config.init_lr*config.lr_decay**epoch
                self.save_loss_history(self.config.plot_dir)
            self.epoch_count = epoch
            print("------- Epoch {:} out of {:}".format(epoch + 1, self.config.n_epochs))
            print("------- learning rate {0}".format(config.lr))
            dev_score, pred, pres = self.run_epoch(sess, train_set, dev_set)

            if dev_score < best_dev:
                best_dev = dev_score
                best_velocity = pred
                best_pres = pres
                perm_dev, Div_U_operator_dev, U_pressure_dev = zip(*dev_set)
                np.savez(save_file, best_velocity=best_velocity, best_pres=best_pres, perm=perm_dev,
                         U_face_operator=Div_U_operator_dev, pressure=U_pressure_dev)
                print("new best norm found {:}".format(best_dev))
                save_path = self.saver.save(sess, self.config.save_model_path)
                print("Model saved in file: %s" % save_path)

    def p_error(self, model_p, actual_p):
        error = np.linalg.norm(model_p-actual_p)/np.linalg.norm(actual_p)
        return error

    def run_epoch(self, sess, train_examples, dev_set, save_every=2):
        save_dir = self.config.model_save_dir
        latest_file = os.path.join(save_dir, 'latest_train_model')
        config = self.config
        num_train = len(train_examples)
        ind = np.arange(num_train)
        np.random.shuffle(ind)
        ind.astype(int)
        batchNum = 1
        num_batch = int(np.ceil(num_train / config.batch_size))
        for i in range(0, num_train, config.batch_size):
            batch_train = [train_examples[i] for i in ind[i:i + config.batch_size]]
            perm_train, Div_U_operator_train, U_pressure_train = zip(*batch_train)

            loss, loss_ratio = self.train_on_batch(sess, perm_train, Div_U_operator_train, U_pressure_train)
            self.recordOutput(loss, batchNum, loss_ratio)

            print("Loss for Batch {:} out of {:} is: {:}".format(batchNum, num_batch, loss))
            batchNum += 1
        # print('-- loss ratio (Div/Pres) for the last batch: {0}'.format(loss_ratio))
        # get the predictions of this model on the last batch
        pred_train, pres_train = self.predict_on_batch(sess, perm_train, Div_U_operator_train)
        # find the training error
        train_error = self.p_error(pres_train, U_pressure_train)
        print('------- traning error: {0}'.format(train_error))
        np.savez(latest_file, pred_train=pred_train, pres_train=pres_train, perm=perm_train,
                 U_face_operator=Div_U_operator_train,
                 pressure=U_pressure_train)
        # save the results every save_every epochs
        n_save = min(config.batch_size, 1)
        if not self.epoch_count % save_every:
            perm_train, Div_U_operator_train, U_pressure_train = zip(*[train_examples[0]])
            pred_train, pres_train = self.predict_on_batch(sess, perm_train, Div_U_operator_train)
            epoch_file = os.path.join(save_dir, 'epoch' + str(self.epoch_count))
            np.savez(epoch_file, pred_train=pred_train, pres_train=pres_train, perm=perm_train,
                     U_face_operator = Div_U_operator_train, pressure = U_pressure_train)
            # TODO: saving only a few predicted values
            # np.savez(epoch_file, pred_train=pred_train, pres_train=pres_train, perm=perm_train,
            #          U_face_operator=Div_U_operator_train, pressure=U_pressure_train)
            # small_p = [U_pressure_train[ss] for ss in range(n_save)]
            # np.savez(epoch_file, pred_train=pred_train[:n_save, :],
            #          pres_train=pres_train[:n_save, :, :, :], pressure=small_p,
            #          U_face_operator=Div_U_operator_train[:n_save])

        # print("------Evaluating on dev set------")
        # batch over dev
        num_dev = len(dev_set)
        batchNum = 1
        ind_dev = np.arange(num_dev)
        num_batch = int(np.ceil(num_dev / config.batch_size))
        norms = []
        predAll = np.zeros((num_dev, self.config.nfaces))
        presAll = np.zeros((num_dev, self.config.nx, self.config.nx, 1))
        for devInd in range(0, num_dev, config.batch_size):
            batch_dev = [dev_set[i] for i in ind_dev[devInd:devInd + config.batch_size]]
            perm_dev, Div_U_operator_dev, U_pressure_dev = zip(*batch_dev)
            pred, pres = self.predict_on_batch(sess, perm_dev, Div_U_operator_dev)
            # calculate norm of pressure
            for j in range(len(U_pressure_dev)):
                norms.append(np.sum(np.power(U_pressure_dev[j] - pres[j, :, :, :], 2)))
            presAll[devInd:devInd + config.batch_size] = pres
            predAll[devInd:devInd + config.batch_size] = pred
            averageNorm = np.sum(norms) * 0.5 / len(norms)
            # print("Norm for Batch {:} out of {:} is: {:}".format(batchNum,num_batch,averageNorm))
            batchNum += 1
        # find the validation error for one batch of validation
        validation_error = self.p_error(pres, U_pressure_dev)
        print('------- validation error: {0}'.format(validation_error))
        if not self.epoch_count % save_every:
            epoch_file = os.path.join(save_dir, 'epoch_dev' + str(self.epoch_count))
            # TODO: saving only a few dev values
            # np.savez(epoch_file, pred_train=pred_train, pres_train=pres_train, perm=perm_train,
            #          U_face_operator=Div_U_operator_train, pressure=U_pressure_train)
            small_p = [U_pressure_dev[ss] for ss in range(n_save)]
            np.savez(epoch_file, pred_train=pred[:n_save, :],
                     pres_train=pres[:n_save, :, :, :], pressure=small_p,
                     U_face_operator=Div_U_operator_dev[:n_save])
        return np.sum(norms) * 0.5 / len(dev_set), predAll, presAll

    def recordOutput(self, loss, batchNum, loss_ratio=None):
        self.loss_history.append(loss)
        self.iter_number.append(
            float(self.epoch_count) + np.float(batchNum * self.config.batch_size / np.float(self.config.n_train)))
        if True:
            self.loss_ratio_history.append(loss_ratio)

    def save_loss_history(self, save_folder):
        np.savez(os.path.join(save_folder, 'loss'), loss=self.loss_history, iter_number=self.iter_number)
        fig, ax = plt.subplots(1, 1)
        ax.plot(self.iter_number, self.loss_history)
        fig.savefig(os.path.join(save_folder, 'loss.png'), format='png')
        ax.set_title('loss')
        plt.close(fig)
        fig, ax = plt.subplots(1, 1)
        ax.plot(self.iter_number[1:], self.loss_ratio_history[1:])
        fig.savefig(os.path.join(save_folder, 'loss_ratio.png'), format='png')
        ax.set_title('p_loss/div_loss')
        plt.close(fig)

    def build(self):
        self.add_placeholders()
        self.pred, self.pres = self.add_prediction_op()
        self.loss, self.loss_ratio = self.add_loss_op(self.pred, self.pres)
        self.train_op = self.add_training_op(self.loss)
        self.saver = tf.train.Saver()
        self.loss_history = []
        self.iter_number = []
        self.loss_ratio_history = []

    def __init__(self, config):
        self.config = config
        self.build()