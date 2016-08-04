# coding:utf-8

import os
import gym
import random
import tensorflow as tf
from threading import Thread
from skimage.color import rgb2gray
from skimage.transform import resize

from network import A3CFF
from agent import Agent
from optimizer import RMSPropApplier

NO_OP_STEPS = 30
ENV_NAME = 'Breakout-v0'
NUM_THREADS = 2
GLOBAL_T_MAX = 10000000
RMSP_ALPHA = 0.99
RMSP_EPSILON = 0.1
SAVE_NETWORK_PATH = 'saved_networks/' + ENV_NAME
SAVE_SUMMARY_PATH = 'summary/' + ENV_NAME
LOAD_NETWORK = False
DISPLAY = True


def load_network(self, sess, saver):
    checkpoint = tf.train.get_checkpoint_state(SAVE_NETWORK_PATH)
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print('Successfully loaded: ' + checkpoint.model_checkpoint_path)
    else:
        print('Training new network...')


def get_initial_state(self, observation, last_observation):
    processed_observation = np.maximum(observation, last_observation)
    processed_observation = resize(rgb2gray(processed_observation), (FRAME_WIDTH, FRAME_HEIGHT))
    state = [processed_observation for _ in range(STATE_LENGTH)]
    return np.stack(state, axis=0)


def actor_learner_thread(thread_index, agent, sess, saver, env):
    global global_t
    global_t = 0

    while global_t < GLOBAL_T_MAX:
        terminal = False
        observation = env.reset()
        for _ in range(random.randint(1, NO_OP_STEPS)):
            last_observation = observation
            observation, _, _, _ = env.step(0)  # Do nothing
        state = get_initial_state(observation, last_observation)

        diff_global_t = agent.process(sess, global_t, env, state, terminal, last_observation)
        global_t += diff_global_t


def main():
    envs = [gym.make(ENV_NAME) for _ in range(NUM_THREADS)]
    num_actions = envs[0].action_space.n

    global_network = A3CFF(num_actions)

    lr_in = tf.placeholder(tf.float32)
    optimizer = RMSPropApplier(learning_rate=lr_in,
                            decay=RMSP_ALPHA,
                            momentum=0.0,
                            epsilon=RMSP_EPSILON)

    agents = []
    for i in range(NUM_THREADS):
        agent = Agent(i, num_actions, global_network, lr_in, optimizer)
        agents.append(agent)

    sess = tf.InteractiveSession()
    saver = tf.train.Saver(global_network.get_vars())
    # summary_placeholders, update_ops, summary_op = setup_summary()
    # summary_writer = tf.train.SummaryWriter(SAVE_SUMMARY_PATH, sess.graph)

    if not os.path.exists(SAVE_NETWORK_PATH):
        os.makedirs(SAVE_NETWORK_PATH)

    sess.run(tf.initialize_all_variables())

    if LOAD_NETWORK:
        load_network(sess, saver)

    actor_learner_threads = []
    for i in range(NUM_THREADS):
        actor_learner_threads.append(Thread(target=actor_learner_thread, args=(i, agent[i], sess, saver, envs[i])))

    for thread in actor_learner_threads:
        thread.start()

    while DISPLAY:
        for env in envs:
            env.render()

    for thread in actor_learner_threads:
        thread.join()


if __name__ == '__main__':
    main()
