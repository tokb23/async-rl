# coding:utf-8

import numpy as np
import tensorflow as tf

from constant import FRAME_WIDTH = 84
from constant import FRAME_HEIGHT = 84
from constant import STATE_LENGTH = 4
from constant import ENTROPY_BETA = 0.01


class Network(object):
    def __init__(self, num_actions):
        self.num_actions = num_actions

    def build_training_op(self):
        self.a = tf.placeholder(tf.int64, [None])
        self.r = tf.placeholder(tf.float32, [None])

        # Convert action to one hot vector
        a_one_hot = tf.one_hot(self.a, self.num_actions, 1.0, 0.0)

        # Avoid NaN by clipping pi when its values become zero
        log_pi = tf.reduce_sum(tf.mul(tf.log(tf.clip_by_value(self.pi, 1e-20, 1.0)), a_one_hot), reduction_indices=1)
        entropy = -tf.reduce_sum(self.pi * tf.log(tf.clip_by_value(self.pi, 1e-20, 1.0)), reduction_indices=1)

        advantage = self.r - self.v

        p_loss = -(log_pi * advantage + ENTROPY_BETA * entropy)
        v_loss = tf.square(advantage)
        self.loss = tf.reduce_mean(p_loss + 0.5 * v_loss)

    def sync_with(self, src_network):
        src_vars = src_network.get_vars()
        dst_vars = self.get_vars()

        sync_op = []
        for src_var, dst_var in zip(src_vars, dst_vars):
            sync_op.append(tf.assign(dst_var, src_var))

        return sync_op

    def fc_weight_variable(self, shape):
        input_channels = shape[0]
        d = 1.0 / np.sqrt(input_channels)
        initial = tf.random_uniform(shape, minval=-d, maxval=d)
        return tf.Variable(initial)

    def fc_bias_variable(self, shape, input_channels):
        d = 1.0 / np.sqrt(input_channels)
        initial = tf.random_uniform(shape, minval=-d, maxval=d)
        return tf.Variable(initial)

    def conv_weight_variable(self, shape):
        filter_w = shape[0]
        filter_h = shape[1]
        input_channels = shape[2]
        d = 1.0 / np.sqrt(input_channels * filter_w * filter_h)
        initial = tf.random_uniform(shape, minval=-d, maxval=d)
        return tf.Variable(initial)

    def conv_bias_variable(self, shape, filter_w, filter_h, input_channels):
        d = 1.0 / np.sqrt(input_channels * filter_w * filter_h)
        initial = tf.random_uniform(shape, minval=-d, maxval=d)
        return tf.Variable(initial)

    def conv2d(self, x, W, stride):
        return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='VALID')


class A3CFF(Network):
    def __init__(self, num_actions):
        Network.__init__(self, num_actions)

        self.W_conv1 = self.conv_weight_variable([8, 8, STATE_LENGTH, 16])
        self.b_conv1 = self.conv_bias_variable([16], 8, 8, 4)

        self.W_conv2 = self.conv_weight_variable([4, 4, 16, 32])
        self.b_conv2 = self.conv_bias_variable([32], 4, 4, 16)

        self.W_fc1 = self.fc_weight_variable([2592, 256])
        self.b_fc1 = self.fc_bias_variable([256], 2592)

        self.W_fc2 = self.fc_weight_variable([256, num_actions])
        self.b_fc2 = self.fc_bias_variable([num_actions], 256)

        self.W_fc3 = self.fc_weight_variable([256, 1])
        self.b_fc3 = self.fc_bias_variable([1], 256)

        self.s = tf.placeholder(tf.float32, [None, FRAME_WIDTH, FRAME_HEIGHT, STATE_LENGTH])

        h_conv1 = tf.nn.relu(self.conv2d(self.s, self.W_conv1, 4) + self.b_conv1)
        h_conv2 = tf.nn.relu(self.conv2d(h_conv1, self.W_conv2, 2) + self.b_conv2)

        h_conv2_flat = tf.reshape(h_conv2, [-1, 2592])
        h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, self.W_fc1) + self.b_fc1)

        self.pi = tf.nn.softmax(tf.matmul(h_fc1, self.W_fc2) + self.b_fc2)
        self.v = tf.matmul(h_fc1, self.W_fc3) + self.b_fc3

    def get_pi(self, sess, state):
        pi_out = sess.run(self.pi, feed_dict={self.s: [state]})
        return pi_out[0]

    def get_v(self, sess, state):
        v_out = sess.run(self.v, feed_dict={self.s: [state]})
        return v_out[0]

    def get_vars(self):
        return [self.W_conv1, self.b_conv1, self.W_conv2, self.b_conv2, self.W_fc1, self.b_fc1, self.W_fc2, self.b_fc2, self.W_fc3, self.b_fc3]
