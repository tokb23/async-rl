# coding:utf-8

import random
import numpy as np
import tensorflow as tf
from skimage.color import rgb2gray
from skimage.transform import resize

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
    def __init__(self, thread_id, num_actions, global_network, lr_input, optimizer):
        self.thread_id = thread_id
        self.lr_input = lr_input

        self.local_network = A3CFF(num_actions)
        self.local_network.build_training_op()

        get_grads = tf.gradients(self.local_network.loss, self.local_network.get_vars())
        grads_and_vars = list(zip(get_grads, global_network.get_vars()))
        self.grads_update = optimizer.apply_gradients(grads_and_vars)

        self.sync_op = self.local_network.sync_with(global_network)

    def get_initial_state(self, observation, last_observation):
        processed_observation = np.maximum(observation, last_observation)
        processed_observation = resize(rgb2gray(processed_observation), (FRAME_WIDTH, FRAME_HEIGHT))
        state = [processed_observation for _ in range(STATE_LENGTH)]
        return np.stack(state, axis=2)

    def get_action(self, sess, state, repeated_action, local_t):
        action = repeated_action

        if local_t % ACTION_INTERVAL == 0:
            pi = self.local_network.get_pi(sess, state)

            # Subtract a tiny value from probabilities in order to avoid 'ValueError: sum(pvals[:-1]) > 1.0' in np.random.multinomial
            pi = pi - np.finfo(np.float32).epsneg

            action = np.nonzero(np.random.multinomial(1, pi))[0][0]

            repeated_action = action

        return action, repeated_action

    def preprocess(self, observation, last_observation):
        processed_observation = np.maximum(observation, last_observation)
        processed_observation = resize(rgb2gray(processed_observation), (FRAME_WIDTH, FRAME_HEIGHT))
        return np.reshape(processed_observation, (FRAME_WIDTH, FRAME_HEIGHT, 1))

    def run(self, sess, state, terminal, state_batch, action_batch, reward_batch, learning_rate, local_t, local_t_start):
        if terminal:
            r = 0
        else:
            r = self.local_network.get_v(sess, state)

        r_batch = np.zeros(local_t - local_t_start)

        for i in reversed(range(local_t_start, local_t)):
            r = reward_batch[i - local_t_start] + GAMMA * r
            r_batch[i - local_t_start] = r

        loss, _ = sess.run([self.local_network.loss, self.grads_update], feed_dict={
            self.local_network.s: state_batch,
            self.local_network.a: action_batch,
            self.local_network.r: r_batch,
            self.lr_input: learning_rate
        })

        return loss

    def actor_learner_thread(self, env, sess, saver):
        global global_t, learning_rate
        global_t = 0
        local_t = 0
        learning_rate = INITIAL_LEARNING_RATE
        lr_step = INITIAL_LEARNING_RATE / GLOBAL_T_MAX
        repeated_action = 0

        total_loss = []

        terminal = False
        observation = env.reset()
        for _ in range(random.randint(1, NO_OP_STEPS)):
            last_observation = observation
            observation, _, _, _ = env.step(0)  # Do nothing
        state = self.get_initial_state(observation, last_observation)

        while global_t < GLOBAL_T_MAX:
            local_t_start = local_t

            state_batch = []
            action_batch = []
            reward_batch = []

            sess.run(self.sync_op)

            while not (terminal or ((local_t - local_t_start) == LOCAL_T_MAX)):
                last_observation = observation

                action, repeated_action = self.get_action(sess, state, repeated_action, local_t)

                observation, reward, terminal, _ = env.step(action)

                state_batch.append(state)
                action_batch.append(action)
                reward = np.clip(reward, -1, 1)
                reward_batch.append(reward)

                processed_observation = self.preprocess(observation, last_observation)
                next_state = np.append(state[:, :, 1:], processed_observation, axis=2)

                local_t += 1
                global_t += 1

                # Anneal learning rate linearly over time
                learning_rate -= lr_step
                if learning_rate < 0.0:
                    learning_rate = 0.0

                state = next_state

            loss = self.run(sess, state, terminal, state_batch, action_batch, reward_batch, learning_rate, local_t, local_t_start)
            total_loss.append(loss)

            if terminal:
                print('THREAD: {0:2d} / LOCAL_TIME: {1:8d} / GLOBAL_TIME: {2:10d} / LEARNING_RATE: {3:.8f} / AVG_LOSS: {4:.5f}'.format(self.thread_id, local_t, global_t, learning_rate, sum(total_loss) / len(total_loss)))

                total_loss = []

                terminal = False
                observation = env.reset()
                for _ in range(random.randint(1, NO_OP_STEPS)):
                    last_observation = observation
                    observation, _, _, _ = env.step(0)  # Do nothing
                state = self.get_initial_state(observation, last_observation)
