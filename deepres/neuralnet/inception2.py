import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
from model import Model
from deepres.neuralnet.base_model import BaseModel
import os as os
import matplotlib.pyplot as plt

class InceptionTwo(BaseModel):
    def __init__(self, config):
        self.config = config
        self.build()

    def inception(self, layer_input):
        config = self.config
        relu = tf.nn.relu
        xavier = tf.contrib.layers.xavier_initializer()
        conv3a_2 = tf.layers.conv2d(layer_input, filters=128, kernel_size=[1, 1], kernel_initializer=xavier, padding="same", activation=relu)
        conv3a_upscaled_2 = tf.image.resize_images(conv3a_2, [config.nx, config.nx])
        conv3b_2 = tf.layers.conv2d(layer_input, filters=96, kernel_size=[1, 1], kernel_initializer=xavier, padding="same", activation=relu)
        conv4b_2 = tf.layers.conv2d(conv3b_2 ,filters=128,kernel_size=[3,3],kernel_initializer=xavier, padding="same",activation=relu)
        conv4b_upscaled_2 = tf.image.resize_images(conv4b_2, [config.nx, config.nx])
        conv3c_2 = tf.layers.conv2d(layer_input, filters=96, kernel_size=[1, 1], kernel_initializer=xavier, padding="same", activation=relu)
        conv4c_2 = tf.layers.conv2d(conv3c_2 ,filters=96,kernel_size=[5,5],kernel_initializer=xavier, padding="same",activation=relu)
        conv4c_upscaled_2 = tf.image.resize_images(conv4c_2, [config.nx, config.nx])
        pool3_2 = tf.layers.max_pooling2d(inputs=layer_input, pool_size=[3, 3], strides=1, padding ='same')
        pool3_conv1_2 = tf.layers.conv2d(pool3_2, filters=96,kernel_size=[1,1],kernel_initializer=xavier, padding="same",activation=relu)
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
        conv2d = tf.layers.conv2d
        conv2d_transpose = tf.contrib.layers.conv2d_transpose
        config = self.config
        n_i = 3
        # add exp of permeability to the input
        input = tf.concat([self.perm_placeholder, tf.exp(self.perm_placeholder)], axis=3)
        # add three convolution sizes to the input
        conv3i = conv2d(input, filters=n_i, kernel_size=[3, 3], kernel_initializer=xavier, padding='same', activation=relu)
        conv5i = conv2d(input, filters=n_i, kernel_size=[5, 5], strides=2, kernel_initializer=xavier, padding='same',
                        activation=relu)
        conv5ia = conv2d_transpose(conv5i, n_i, [5,5], 2)
        conv7i = conv2d(input, filters=n_i, kernel_size=[7, 7], strides=2, kernel_initializer=xavier, padding='same',
                        activation=relu)
        conv7ia = conv2d_transpose(conv7i, n_i, [7, 7], 2)
        augmented_input = tf.concat([input, conv3i, conv5ia, conv7ia], axis=3)

        conv1_7x7_s2 = tf.layers.conv2d(augmented_input, filters=64,kernel_size=[7,7],kernel_initializer=xavier, padding="same",activation=relu)
        pool1_3x3_s2 = tf.layers.max_pooling2d(inputs=conv1_7x7_s2, pool_size=[3,3], strides=2, padding = 'same')
        pool1_norm1 = tf.nn.lrn(pool1_3x3_s2)
        conv2_3x3_reduce = tf.layers.conv2d(pool1_norm1, filters=96,kernel_size=[1,1],kernel_initializer=xavier, padding="same",activation=relu)
        conv2_3x3 = tf.layers.conv2d(conv2_3x3_reduce ,filters=96,kernel_size=[3,3],kernel_initializer=xavier, padding="same",activation=relu)
        conv2_norm2 = tf.nn.lrn(conv2_3x3)
        pool2_3x3_s2 = tf.layers.max_pooling2d(inputs=conv2_norm2, pool_size=[3,3], strides=2, padding = 'same')

        conv3a = tf.layers.conv2d(pool2_3x3_s2, filters=96,kernel_size=[1,1],kernel_initializer=xavier, padding="same",activation=relu)
        conv3a_upscaled = tf.image.resize_images(conv3a, [config.nx, config.nx])
        conv3b = tf.layers.conv2d(pool2_3x3_s2 ,filters=96,kernel_size=[1,1],kernel_initializer=xavier, padding="same",activation=relu)
        conv4b = tf.layers.conv2d(conv3b ,filters=96,kernel_size=[3,3],kernel_initializer=xavier, padding="same",activation=relu)
        conv4b_upscaled = tf.image.resize_images(conv4b, [config.nx, config.nx])
        conv3c = tf.layers.conv2d(pool2_3x3_s2, filters=96,kernel_size=[1,1],kernel_initializer=xavier, padding="same",activation=relu)
        conv4c = tf.layers.conv2d(conv3c ,filters=96,kernel_size=[5,5],kernel_initializer=xavier, padding="same",activation=relu)
        conv4c_upscaled = tf.image.resize_images(conv4c, [config.nx, config.nx])
        pool3 = tf.layers.max_pooling2d(inputs=pool2_3x3_s2, pool_size=[3,3], strides=1, padding = 'same')
        pool3_conv1 = tf.layers.conv2d(pool3, filters=96,kernel_size=[1,1],kernel_initializer=xavier, padding="same",activation=relu)
        conv2_pool3_upscaled = tf.image.resize_images(pool3_conv1, [config.nx, config.nx])
        inception1 = relu(tf.concat([conv1_7x7_s2,conv3a_upscaled,conv4b_upscaled,conv4c_upscaled,conv2_pool3_upscaled], axis=3))
        last_inception = inception1
        for i in range(config.n_inception-1):
            last_inception = self.inception(last_inception)
        last_inception = tf.layers.dropout(last_inception, rate=config.dropout, training=self.is_training)
        inception_final_conv1 = tf.layers.conv2d(last_inception ,filters=128,kernel_size=[3,3],kernel_initializer=xavier, padding="same",activation=relu)
        inception_final_conv2 = tf.layers.conv2d(inception_final_conv1 ,filters=192,kernel_size=[1,1],kernel_initializer=xavier, padding="same",activation=relu)

        pressure = tf.layers.conv2d(inputs=inception_final_conv2, filters=1,kernel_size=[1,1],kernel_initializer=xavier, padding="same")
        pres_flat = tf.reshape(pressure,[-1,config.nx*config.nx,1])*config.max_val + config.mean_val
        dense_operator = tf.sparse_tensor_to_dense(tf.sparse_reorder(self.Div_U_operator_placeholder))
        Divergence = tf.matmul(dense_operator, pres_flat)
        Divergence = tf.reshape(Divergence,[-1,config.nfaces])
        return Divergence, pressure

    def add_loss_op(self, pred_divergence, pred_pressure):
        """Adds Ops for the loss function to the computational graph.
        Args:
            pred: A tensor of shape (batch_size, n_classes)
        Returns:
            loss: A 0-d tensor (scalar) output
        """
        config = self.config
        div_weight, tv_weight = config.weight, config.tv_weight
        lp = tf.nn.l2_loss(pred_pressure - self.pressure_placeholder)
        # ld = div_weight * tf.nn.l2_loss(pred_divergence)
        # ld = div_weight * tf.nn.l
        ld = div_weight * tf.reduce_mean(tf.norm(pred_divergence, ord=1))
        loss_ratio = lp/ld
        # # total variation of divergence
        # div_tensor = tf.reshape(pred_divergence, [-1, config.nx, config.nx, 1])
        # lv = tv_weight * tf.reduce_mean(tf.image.total_variation(div_tensor))
        # set lv to the second norm
        lv = tv_weight * tf.reduce_mean(tf.norm(pred_divergence, ord=2))
        loss =  lp + ld + lv
        return loss, loss_ratio
