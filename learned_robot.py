"""
DDPG Agent learning example using Keras-RL.
"""

import os
import pickle

from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Concatenate, Input, Dropout
from keras.optimizers import Adam, RMSprop
from keras.callbacks import TensorBoard
from rl.agents import DDPGAgent
from rl.random import OrnsteinUhlenbeckProcess
from rl.memory import SequentialMemory
from rl.callbacks import ModelIntervalCheckpoint
import gym
import numpy as np
from tqdm import tqdm

import gym_line_follower  # to register environment

# Number of past subsequent observations to take as input
window_length = 5


def build_agent(env):
    nb_actions = env.action_space.shape[0]
    print(env.observation_space.shape)

    # Actor model
    actor = Sequential()
    actor.add(Flatten(input_shape=(window_length,) + env.observation_space.shape))
    actor.add(Dense(256, activation="relu"))
    actor.add(Dense(256, activation="relu"))
    actor.add(Dense(128, activation="relu"))
    actor.add(Dense(nb_actions, activation="tanh"))
    actor.summary()

    # Critic model
    action_input = Input(shape=(nb_actions,), name='action_input')
    observation_input = Input(shape=(window_length,) + env.observation_space.shape, name='observation_input')
    flattened_observation = Flatten()(observation_input)
    x = Concatenate()([action_input, flattened_observation])
    x = Dense(256, activation="relu")(x)
    x = Dense(512, activation="relu")(x)
    x = Dense(256, activation="relu")(x)
    x = Dense(1, activation="linear")(x)
    critic = Model(inputs=[action_input, observation_input], outputs=x)
    critic.summary()

    memory = SequentialMemory(limit=1000000, window_length=window_length)
    # Exploration policy - has a great effect on learning. Should encourage forward motion.
    # theta - how fast the process returns to the mean
    # mu - mean value - this should be greater than 0 to encourage forward motion
    # sigma - volatility of the process
    random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.5, mu=0.5, sigma=0.5)
    agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                      memory=memory, nb_steps_warmup_critic=80, nb_steps_warmup_actor=80,
                      random_process=random_process, gamma=.99, target_model_update=1e-3)
    agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])
    return agent

def test(env, path):
    agent = build_agent(env)
    agent.load_weights(path)
    agent.test(env, nb_episodes=25, visualize=False)

if __name__ == '__main__':
    # Use gym.make with custom arguments
    all_labels = [
        np.array([0.9, 0.8, 1.0, 0.9, 0.9, 0.9]),
        np.array([1.0, 0.9, 0.8, 0.9, 0.9, 0.8]),
        np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        np.array([1.0, 0.9, 0.8, 0.9, 0.9, 0.8]),
    ]

    for _ in range(45):
        random_sample = np.random.choice([1.0, 0.9, 0.8, 0.0], 6, p=[0.3, 0.3, 0.2, 0.2])
        all_labels.append(random_sample)



    for label in tqdm(all_labels):
        custom_env_args = {
            'gui': False,
            'hardware_label': label
        }
        env = gym.make("LineFollower-v0", **custom_env_args)
        test(env, "models/ddpg_2/last_weights.h5f")

    