#############################################
# Copyright (c) 2018-present
# written by Kai Wu on 2018-08-04
#############################################

"""This is implementation of AlexNet model."""

import tensorflow as tf
from nn_layers import conv_bn_relu, max_pool, dense, flatten


class AlexNet(object):

    def __init__(self, x, output_dim, keep_prob):

        self.scores = AlexNet.forward(x, output_dim, keep_prob)

    @staticmethod
    def forward(x, output_dim, keep_prob):
        layers = []

        with tf.variable_scope('conv1'):
            conv1 = conv_bn_relu(x, [3, 3, 3, 24], 1)
            pool1 = max_pool(conv1, ksize=2, stride=2, padding='SAME')
            layers.append(pool1)

        with tf.variable_scope('conv2'):
            conv2 = conv_bn_relu(layers[-1], [3, 3, 24, 96], 1)
            pool2 = max_pool(conv2, ksize=2, stride=2, padding='SAME')
            layers.append(pool2)

        with tf.variable_scope('conv3'):
            conv3 = conv_bn_relu(layers[-1], [3, 3, 96, 192], 1)
            layers.append(conv3)

        with tf.variable_scope('conv4'):
            conv4 = conv_bn_relu(layers[-1], [3, 3, 192, 192], 1)
            layers.append(conv4)

        with tf.variable_scope('conv5'):
            conv5 = conv_bn_relu(layers[-1], [3, 3, 192, 96], 1)
            pool5 = max_pool(conv5, ksize=2, stride=2, padding='SAME')
            layers.append(pool5)

        with tf.variable_scope('fc6'):
            flattened = flatten(layers[-1])
            fc6 = dense(flattened, 4096, use_relu=True)
            dropout6 = tf.nn.dropout(fc6, keep_prob)
            layers.append(dropout6)

        with tf.variable_scope('fc7'):
            fc7 = dense(layers[-1], 4096, use_relu=True)
            dropout7 = tf.nn.dropout(fc7, keep_prob)
            layers.append(dropout7)

        with tf.variable_scope('softmax'):
            scores = dense(layers[-1], output_dim, use_relu=False)
            layers.append(scores)

        return layers[-1]

