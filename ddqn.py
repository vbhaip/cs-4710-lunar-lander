# This runs a ddqn model with a keras neural network -- difference between this and dqn.py is that this uses a more sophisticated architecture
# https://github.com/taylormcnally/keras-rl2/blob/master/examples/dqn_cartpole.py
import numpy as np
import gym
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory


ENV_NAME = 'LunarLander-v2'


# set up environment
env = gym.make(ENV_NAME)
np.random.seed(4710)
env.seed(4710)
nb_actions = env.action_space.n
layer_size = 16


load_weights = True

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
memory = SequentialMemory(limit=500000, window_length=1)
policy = BoltzmannQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
               target_model_update=1e-2, policy=policy, enable_double_dqn=True)
dqn.compile(Adam(learning_rate=1e-3), metrics=['mae'])


# test to see how it does beforehand
#dqn.test(env, nb_episodes=5, visualize=True)

if(load_weights):
    #if weights exist then load them and test
    dqn.load_weights(f"ddqn_{ENV_NAME}_weights.h5f")
    history = dqn.test(env, nb_episodes=50, visualize=False)
    plt.plot(history.history['episode_reward'])
    plt.title('reward over time')
    plt.ylabel('reward')
    plt.xlabel('epoch')
    plt.savefig('dqn_test_results.png')
    print("average score: ", np.mean(history.history['episode_reward']))
else:
    history = dqn.fit(env, nb_steps=200000, visualize=False, verbose=2)
    plt.plot(history.history['episode_reward'])
    plt.title('reward over time')
    plt.ylabel('reward')
    plt.xlabel('epoch')
    plt.savefig('dqn_training_results.png')
    dqn.save_weights(f'ddqn_{ENV_NAME}_weights.h5f', overwrite=True)

    # run dqn test
    history = dqn.test(env, nb_episodes=50, visualize=False)

    #print average reward
    print(sum(history.history['episode_reward'])/50.)