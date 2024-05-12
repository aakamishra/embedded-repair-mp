"""
DDPG Agent learning example using Keras-RL.
"""

import os
import pickle
import numpy as np

from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Concatenate, Input, Dropout
from keras.optimizers import Adam, RMSprop
from keras.callbacks import TensorBoard
from rl.agents import DDPGAgent
from rl.random import OrnsteinUhlenbeckProcess
from rl.memory import SequentialMemory
from rl.callbacks import ModelIntervalCheckpoint
import gym

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
    agent.compile(Adam(lr=.0001, clipnorm=1.), metrics=['mae'])
    return agent


def train(env, name, steps=25000, pretrained_path=None):
    agent = build_agent(env)
    # Load pre-trained weights optionally
    if pretrained_path is not None:
        agent.load_weights(pretrained_path)

    save_path = os.path.join("models", name)
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(os.path.join(save_path, "checkpoints"), exist_ok=True)
    h = agent.fit(env, nb_steps=steps, visualize=False, verbose=2,
                  callbacks=[ModelIntervalCheckpoint(os.path.join(save_path, "checkpoints", "chkpt_{step}.h5f"),
                                                     interval=int(steps/20), verbose=1),
                             TensorBoard(log_dir=os.path.join("logs", name))])

    pickle.dump(h.history, open(os.path.join(save_path, "history.pkl"), "wb"))

    agent.save_weights(os.path.join(save_path, "last_weights.h5f"), overwrite=True)


def test(env, path):
    agent = build_agent(env)
    agent.load_weights(path)
    vals = agent.test(env, nb_episodes=1, visualize=False)
    print("reward history", vals.history["episode_reward"][0])
    print("avg nb steps", vals.history["nb_steps"][0])
    return vals.history["episode_reward"][0], vals.history["nb_steps"][0]


if __name__ == '__main__':
    custom_env_args = {
        'gui': False,
        'hardware_label': np.ones(4),
        'model_path': None, 
        'data_collection_file': None,
        'number_of_wheels': 4,
        'simulated_firmware_file':"/Users/aakamishra/school/cs329m/embedded-repair-mp/gym_line_follower/line_follower_bot_four_wheel_reference.py",
        'urdf_file':'four_wheel_robot.urdf',
    }
    env = gym.make("LineFollower-v0", **custom_env_args)
    train(env, "ddpg_four_wheels_test", steps=200000, pretrained_path=None)
    rewards = []
    steps = []
    for i in range(50):
        reward, step = test(env, "models/ddpg_four_wheels_test/last_weights.h5f")
        rewards.append(reward)
        steps.append(step)
    print(np.mean(rewards), np.mean(steps))