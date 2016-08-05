# coding:utf-8

import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize
import tensorflow as tf

from network import A3CFF

FRAME_WIDTH = 84
FRAME_HEIGHT = 84
STATE_LENGTH = 4
GAMMA = 0.99
LOCAL_T_MAX = 5
ENTROPY_BETA = 0.01
INITIAL_LEARNING_RATE = 0.007
GLOBAL_T_MAX = 10000000
NO_OP_STEPS = 30
ACTION_INTERVAL = 4


class Agent(object):
    def __init__(self, thread_index, num_actions, global_network, lr_in, optimizer):
        self.thread_index = thread_index
        self.lr_in = lr_in

        self.local_network = A3CFF(num_actions)
        self.local_network.build_training_op()

        get_gradients = tf.gradients(self.local_network.loss, self.local_network.get_vars())
        grads_and_vars = list(zip(get_gradients, global_network.get_vars()))
        self.grad_update = optimizer.apply_gradients(grads_and_vars)

        self.sync = self.local_network.sync_from(global_network)

        self.local_t = 0

        self.repeated_action = 0

        self.learning_rate = INITIAL_LEARNING_RATE
        self.lr_step = INITIAL_LEARNING_RATE / GLOBAL_T_MAX

        self.episode_reward = 0

    def get_action(self, pi):
        action = self.repeated_action

        if self.local_t % ACTION_INTERVAL == 0:
            # Subtract a tiny value from probabilities in order to avoid 'ValueError: sum(pvals[:-1]) > 1.0' in np.random.multinomial
            probs = pi - np.finfo(np.float32).epsneg

            action = np.nonzero(np.random.multinomial(1, probs))[0][0]

            self.repeated_action = action

        return action

    def preprocess(self, observation, last_observation):
        processed_observation = np.maximum(observation, last_observation)
        processed_observation = resize(rgb2gray(processed_observation), (FRAME_WIDTH, FRAME_HEIGHT))
        return np.reshape(processed_observation, (FRAME_WIDTH, FRAME_HEIGHT, 1))

    def process(self, sess, global_t, env, state, terminal, last_observation):
        states = []
        actions = []
        rewards = []

        sess.run(self.sync)

        self.local_t_start = self.local_t

        while not (terminal or ((self.local_t - self.local_t_start) == LOCAL_T_MAX)):
            pi_ = self.local_network.get_policy(sess, state)
            action = self.get_action(pi_)

            observation, reward, terminal, _ = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(np.clip(reward, -1, 1))

            processed_observation = self.preprocess(observation, last_observation)
            next_state = np.append(state[:, :, 1:], processed_observation, axis=2)

            self.local_t += 1

            self.learning_rate -= self.lr_step
            if self.learning_rate < 0.0:
                self.learning_rate = 0.0

            state = next_state

        if terminal:
            R = 0
        else:
            R = self.local_network.get_value(sess, state)

        R_batch = np.zeros(self.local_t - self.local_t_start)

        for i in reversed(range(self.local_t_start, self.local_t)):
            R = rewards[i - self.local_t_start] + GAMMA * R
            R_batch[i - self.local_t_start] = R

        sess.run(self.grad_update, feed_dict={
            self.local_network.s: states,
            self.local_network.a: actions,
            self.local_network.r: R_batch,
            self.lr_in: self.learning_rate})

        if self.local_t % 100 == 0:
            print('thread_id: {0} | local_t: {1}'.format(self.thread_index, self.local_t))

        diff_local_t = self.local_t - self.local_t_start

        return diff_local_t
