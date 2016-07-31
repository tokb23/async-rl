# coding:utf-8

import gym
import random
import numpy as np
import tensorflow as tf
from threading import Thread
from skimage.color import rgb2gray
from skimage.transform import resize
from keras.models import Model
from keras.layers import Convolution2D, Flatten, Dense, Input

ENV_NAME = 'Breakout-v0'  # Environment name
FRAME_WIDTH = 84  # Resized frame width
FRAME_HEIGHT = 84  # Resized frame height
STATE_LENGTH = 4  # Number of most recent frames to produce the input to the network
LEARNING_RATE = 0.0007  # Learning rate used by RMSProp
DECAY = 0.99  # decay factor used by RMSProp
MOMENTUM = 0.95  # Momentum used by RMSProp
MIN_GRAD = 0.1  # Constant added to the squared gradient in the denominator of the RMSProp update
NO_OP_STEPS = 30  # Maximum number of "do nothing" actions to be performed by the agent at the start of an episode
ACTION_INTERVAL = 4  # The agent sees only every 4th input
GAMMA = 0.99  # Discount factor
# ENTROPY_BETA = 0.01
NUM_THREADS = 2  # Number of thread
GLOBAL_T_MAX = 320000000
THREAD_T_MAX = 5
TRAIN = True


class Agent():
    def __init__(self, num_actions):
        self.num_actions = num_actions
        self.repeated_action = 0

        # Create policy and value networks
        self.s, self.action_probs, self.state_value = self.build_networks()

        # Define loss and gradient update operation
        self.a, self.r, self.grad_update = self.build_training_op()

        self.sess = tf.InteractiveSession()

        self.sess.run(tf.initialize_all_variables())

    def build_networks(self):
        s_in = Input(shape=(STATE_LENGTH, FRAME_WIDTH, FRAME_HEIGHT))
        shared = Convolution2D(16, 8, 8, subsample=(4, 4), activation='relu')(s_in)
        shared = Convolution2D(32, 4, 4, subsample=(2, 2), activation='relu')(shared)
        shared = Flatten()(shared)
        shared = Dense(256, activation='relu')(shared)
        p_out = Dense(self.num_actions, activation='softmax')(shared)
        v_out = Dense(1)(shared)

        policy_network = Model(input=s_in, output=p_out)
        value_network = Model(input=s_in, output=v_out)

        s = tf.placeholder(tf.float32, [None, STATE_LENGTH, FRAME_WIDTH, FRAME_HEIGHT])
        action_probs = policy_network(s)
        state_value = value_network(s)

        return s, action_probs, state_value

    def build_training_op(self):
        a = tf.placeholder(tf.int64, [None])
        r = tf.placeholder(tf.float32, [None])

        # Convert action to one hot vector
        a_one_hot = tf.one_hot(a, self.num_actions, 1.0, 0.0)
        log_prob = tf.log(tf.reduce_sum(tf.mul(self.action_probs, a_one_hot), reduction_indices=1))
        # entropy = tf.reduce_sum(self.action_probs * tf.log(self.action_probs), reduction_indices=1)

        # p_loss = -(log_prob * (r - self.state_value) + ENTROPY_BETA * entropy)
        p_loss = -log_prob * (r - self.state_value)
        v_loss = tf.reduce_mean(tf.square(r - self.state_value))
        loss = p_loss + 0.5 * v_loss

        optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE, decay=DECAY, momentum=MOMENTUM, epsilon=MIN_GRAD)
        grad_update = optimizer.minimize(loss)

        return a, r, grad_update

    def get_initial_state(self, observation, last_observation):
        processed_observation = np.maximum(observation, last_observation)
        processed_observation = resize(rgb2gray(processed_observation), (FRAME_WIDTH, FRAME_HEIGHT))
        state = [processed_observation for _ in range(STATE_LENGTH)]
        return np.stack(state, axis=0)

    def get_action(self, state, t):
        action = self.repeated_action

        if (t - 1) % ACTION_INTERVAL == 0:
            probs = self.sess.run(self.action_probs, feed_dict={self.s: [state]})[0]

            # Subtract a tiny value from probabilities in order to avoid 'ValueError: sum(pvals[:-1]) > 1.0' in np.random.multinomial
            probs = probs - np.finfo(np.float32).epsneg

            action = np.nonzero(np.random.multinomial(1, probs))[0][0]

            self.repeated_action = action

        return action

    def run(self, state, terminal, t, t_start, state_batch, action_batch, reward_batch):
        if terminal:
            R = 0
        else:
            R = self.sess.run(self.state_value, feed_dict={self.s: [state]})[0]

        R_batch = np.zeros(t - t_start)

        for i in reversed(range(t_start, t)):
            R = reward_batch[i - t_start] + GAMMA * R
            R_batch[i - t_start] = R

        self.sess.run(self.grad_update, feed_dict={
            self.s: state_batch,
            self.a: action_batch,
            self.r: R_batch
        })


def actor_learner_thread(thread_id, env, agent):
    global T
    T = 0
    t = 1

    terminal = False
    observation = env.reset()
    for _ in range(random.randint(1, NO_OP_STEPS)):
        last_observation = observation
        observation, _, _, _ = env.step(0)  # Do nothing
    state = agent.get_initial_state(observation, last_observation)

    while T < GLOBAL_T_MAX:
        t_start = t

        state_batch = []
        action_batch = []
        reward_batch = []

        while not (terminal or ((t - t_start) == THREAD_T_MAX)):
            last_observation = observation

            action = agent.get_action(state, t)

            observation, reward, terminal, _ = env.step(action)

            state_batch.append(state)
            action_batch.append(action)
            reward = np.clip(reward, -1, 1)
            reward_batch.append(reward)

            processed_observation = preprocess(observation, last_observation)
            next_state = np.append(state[1:, :, :], processed_observation, axis=0)

            t += 1
            T += 1

            state = next_state

        agent.run(state, terminal, t, t_start, state_batch, action_batch, reward_batch)

        if terminal:
            # Debug
            print('THREAD: {0} / TIMESTEP: {1} / GLOBAL_TIME {2}'.format(thread_id, t, T))

            terminal = False
            observation = env.reset()
            for _ in range(random.randint(1, NO_OP_STEPS)):
                last_observation = observation
                observation, _, _, _ = env.step(0)  # Do nothing
            state = agent.get_initial_state(observation, last_observation)


def preprocess(observation, last_observation):
    processed_observation = np.maximum(observation, last_observation)
    processed_observation = resize(rgb2gray(processed_observation), (FRAME_WIDTH, FRAME_HEIGHT))
    return np.reshape(processed_observation, (1, FRAME_WIDTH, FRAME_HEIGHT))


def main():
    envs = [gym.make(ENV_NAME) for _ in range(NUM_THREADS)]
    agent = Agent(num_actions=envs[0].action_space.n)

    if TRAIN:
        actor_learner_threads = [Thread(target=actor_learner_thread, args=(i, envs[i], agent)) for i in range(NUM_THREADS)]

        for thread in actor_learner_threads:
            thread.start()

        while True:
            for env in envs:
                env.render()

        for thread in actor_learner_threads:
            thread.join()


if __name__ == '__main__':
    main()
