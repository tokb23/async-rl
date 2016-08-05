# coding:utf-8

import os
import gym
import tensorflow as tf
from threading import Thread

from network import A3CFF
from agent import Agent

FRAME_WIDTH = 84
FRAME_HEIGHT = 84
STATE_LENGTH = 4
NO_OP_STEPS = 30
ENV_NAME = 'Breakout-v0'
NUM_THREADS = 2
GLOBAL_T_MAX = 10000000
RMSP_ALPHA = 0.99
RMSP_EPSILON = 0.1
SAVE_NETWORK_PATH = 'saved_networks/' + ENV_NAME
LOAD_NETWORK = False
DISPLAY = False


def load_network(sess, saver):
    checkpoint = tf.train.get_checkpoint_state(SAVE_NETWORK_PATH)
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print('Successfully loaded: ' + checkpoint.model_checkpoint_path)
    else:
        print('Training new network...')


def main():
    envs = [gym.make(ENV_NAME) for _ in range(NUM_THREADS)]

    num_actions = envs[0].action_space.n
    global_network = A3CFF(num_actions)

    lr_input = tf.placeholder(tf.float32)
    optimizer = tf.train.RMSPropOptimizer(lr_input, decay=RMSP_ALPHA, epsilon=RMSP_EPSILON)

    agents = [Agent(i, num_actions, global_network, lr_input, optimizer) for i in range(NUM_THREADS)]

    sess = tf.InteractiveSession()
    saver = tf.train.Saver(global_network.get_vars())

    if not os.path.exists(SAVE_NETWORK_PATH):
        os.makedirs(SAVE_NETWORK_PATH)

    sess.run(tf.initialize_all_variables())

    if LOAD_NETWORK:
        load_network(sess, saver)

    actor_learner_threads = []
    for i in range(NUM_THREADS):
        env = envs[i]
        agent = agents[i]
        actor_learner_threads.append(Thread(target=agent.actor_learner_thread, args=(env, sess, saver)))

    for thread in actor_learner_threads:
        thread.start()

    while DISPLAY:
        for env in envs:
            env.render()

    for thread in actor_learner_threads:
        thread.join()


if __name__ == '__main__':
    main()
