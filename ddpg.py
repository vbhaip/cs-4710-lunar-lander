# This was a previous file when experimenting to try to solve another environment

# https://github.com/taylormcnally/keras-rl2/blob/master/examples/ddpg_pendulum.py
# deep deterministic policy gradient -- combines policy gradient and deep q network
import numpy as np
import gym

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Flatten, Input, Concatenate
from tensorflow.keras.optimizers import Adam

from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess


ENV_NAME = 'BipedalWalker-v3'


# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
np.random.seed(4710)
env.seed(4710)

# assert len(env.action_space.shape) == 1
nb_actions = env.action_space.shape[0]

#model
actor = Sequential()
actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
actor.add(Dense(16))
actor.add(Activation('relu'))
actor.add(Dense(16))
actor.add(Activation('relu'))
actor.add(Dense(16))
actor.add(Activation('relu'))
actor.add(Dense(nb_actions))
actor.add(Activation('linear'))
print(actor.summary())


action_input = Input(shape=(nb_actions,), name='action_input')
observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
flattened_observation = Flatten()(observation_input)
x = Concatenate()([action_input, flattened_observation])
x = Dense(32)(x)
x = Activation('relu')(x)
x = Dense(32)(x)
x = Activation('relu')(x)
x = Dense(32)(x)
x = Activation('relu')(x)
x = Dense(1)(x)
x = Activation('linear')(x)
critic = Model(inputs=[action_input, observation_input], outputs=x)
print(critic.summary())

# create agent with hyperparams
memory = SequentialMemory(limit=100000, window_length=1)
random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.15, mu=0., sigma=.3)
agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                  memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
                  random_process=random_process, gamma=.99, target_model_update=1e-3)
agent.compile(Adam(learning_rate=.001, clipnorm=1.), metrics=['mae'])

# change visualize=False to speed up training
agent.fit(env, nb_steps=50000, visualize=False, verbose=1, nb_max_episode_steps=200)

# save weights
agent.save_weights(f'ddpg_{ENV_NAME}_weights.h5f', overwrite=True)

# run on 5 episodes for testing
agent.test(env, nb_episodes=5, visualize=True, nb_max_episode_steps=200)