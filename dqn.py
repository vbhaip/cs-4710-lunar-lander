# This runs a dqn model with a keras neural network
# https://github.com/taylormcnally/keras-rl2/blob/master/examples/dqn_cartpole.py
import numpy as np
import gym

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory


#set the problem name
ENV_NAME = 'LunarLander-v2'


# set up environment
env = gym.make(ENV_NAME)
np.random.seed(4710)
env.seed(4710)
nb_actions = env.action_space.n

#set the layer_size to determine complexity of model
layer_size = 16

# nn for the model
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(layer_size))
model.add(Activation('relu'))
model.add(Dense(layer_size))
model.add(Activation('relu'))
model.add(Dense(layer_size))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())

# compile agent and keep track of past memory
memory = SequentialMemory(limit=50000, window_length=1)

#boltzmann policy = softmax essentially, deals with exploration exploitation problem
policy = BoltzmannQPolicy()

#runs the actual dqn agent
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
               target_model_update=1e-2, policy=policy)

#uses adam optimizer (standard optimization technique)
dqn.compile(Adam(learning_rate=1e-3), metrics=['mae'])


# test to see how it does beforehand
dqn.test(env, nb_episodes=5, visualize=True)

try:
    #if weights exist then load them
    dqn.load_weights(f"dqn_{ENV_NAME}_weights.h5f")
except:
    # if weights don't exist, then train the model
    dqn.fit(env, nb_steps=50000, visualize=False, verbose=2)

    # After training is done, we save the final weights.
    dqn.save_weights(f'dqn_{ENV_NAME}_weights.h5f', overwrite=True)

# run dqn test
dqn.test(env, nb_episodes=50, visualize=True)