# coding:utf-8

import random
import numpy as np
import tensorflow as tf
from skimage.color import rgb2gray
from skimage.transform import resize

from network import A3CFF

from constant import ENV_NAME
from constant import FRAME_WIDTH
from constant import FRAME_HEIGHT
from constant import STATE_LENGTH
from constant import GLOBAL_T_MAX
from constant import LOCAL_T_MAX
from constant import GAMMA
from constant import INITIAL_LEARNING_RATE
from constant import ACTION_INTERVAL
from constant import NO_OP_STEPS
from constant import SAVE_INTERVAL
from constant import SAVE_NETWORK_PATH


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

            histogram = np.random.multinomial(1, pi)
            action = int(np.nonzero(histogram)[0])

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

    def save_network(self, sess, saver, global_t):
        save_path = saver.save(sess, SAVE_NETWORK_PATH + '/' + ENV_NAME, global_step=global_t)
        print('Successfully saved: ' + save_path)

    def write_summary(self, sess, total_reward, duration, global_episode, total_loss, summary_placeholders, update_ops, summary_op, summary_writer):
        stats = [total_reward, duration, sum(total_loss) / len(total_loss)]
        for i in range(len(stats)):
            sess.run(update_ops[i], feed_dict={
                summary_placeholders[i]: float(stats[i])
            })
        summary_str = sess.run(summary_op)
        summary_writer.add_summary(summary_str, global_episode + 1)

    def actor_learner_thread(self, env, sess, saver, summary_placeholders, update_ops, summary_op, summary_writer):
        global global_t, learning_rate, global_episode
        global_t = 0
        local_t = 0
        learning_rate = INITIAL_LEARNING_RATE
        lr_step = INITIAL_LEARNING_RATE / GLOBAL_T_MAX
        repeated_action = 0

        total_reward = 0
        total_loss = []
        duration = 0
        global_episode = 0
        local_episode = 0

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

                total_reward += reward
                duration += 1

                # Anneal learning rate linearly over time
                learning_rate -= lr_step
                if learning_rate < 0.0:
                    learning_rate = 0.0

                state = next_state

            loss = self.run(sess, state, terminal, state_batch, action_batch, reward_batch, learning_rate, local_t, local_t_start)
            total_loss.append(loss)

            if terminal:
                if self.thread_id == 0:
                    # Write summary
                    self.write_summary(sess, total_reward, duration, global_episode, total_loss, summary_placeholders, update_ops, summary_op, summary_writer)

                # Debug
                print('THREAD: {0:2d} / GLOBAL_EPISODE: {1:6d} / GLOBAL_TIME: {2:10d} / LOCAL_EPISODE: {3:4d} / LOCAL_TIME: {4:8d} / DURATION: {5:5d} / TOTAL_REWARD: {6:3.0f} / AVG_LOSS: {7:.5f} / LEARNING_RATE: {8:.10f}'.format(
                    self.thread_id, global_episode + 1, global_t, local_episode + 1, local_t, duration, total_reward, sum(total_loss) / len(total_loss), learning_rate))

                total_reward = 0
                total_loss = []
                duration = 0
                local_episode += 1
                global_episode += 1

                terminal = False
                observation = env.reset()
                for _ in range(random.randint(1, NO_OP_STEPS)):
                    last_observation = observation
                    observation, _, _, _ = env.step(0)  # Do nothing
                state = self.get_initial_state(observation, last_observation)

            # Save network
            if global_t % SAVE_INTERVAL == 0:
                self.save_network(sess, saver, global_t)
