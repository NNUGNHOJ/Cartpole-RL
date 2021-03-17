import gym
import time, math, random
import numpy as np
import tensorflow as tf
import torch
from torch.autograd import Variable
from tensorflow import keras
from sklearn.preprocessing import KBinsDiscretizer
from typing import Tuple


class DQLAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=64, l_rate=0.001):
        self.criterion = torch.nn.MSELoss()

        self.model = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_dim),                
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim*2),   
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim*2, action_dim))
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), l_rate)

    # TODO: annotate
    def update(self, state, y):
        """Update the weights of the network given a training sample. """
        y_pred = self.model(torch.Tensor(state))
        loss = self.criterion(y_pred, Variable(torch.Tensor(y)))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    # TODO: annotate
    def predict(self, state):
        """ Compute Q values for all actions using the DQL. """
        with torch.no_grad():
            return self.model(torch.Tensor(state))

env = gym.make('CartPole-v1')

print("Action Space: {}".format(env.action_space))
print("State space: {}".format(env.observation_space))

"""Discretize the environment by turning the CartPole continuous space
into a single discrete space."""

# n_bins = (24, 6)
# lower_bounds = [env.observation_space.low[2]/3, -1]
# upper_bounds = [env.observation_space.high[2]/3, 1]


# def discretizer(position, __, angle, pole_velocity) -> Tuple[int, ...]:
#     est = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="uniform")
#     est.fit([lower_bounds, upper_bounds])
#     return tuple(map(int, est.transform([[angle, (pole_velocity)]])[0]))

"""Training..."""
epsilon = 0.1
n_episodes = 10000
eps_decay = 0.99
gamma=.9
# Number of states
n_state = env.observation_space.shape[0]
# Number of actions
n_action = env.action_space.n
model = DQLAgent(n_state, n_action)
for e in range(n_episodes):

    state = env.reset()
    done = False
    
    # update network every n steps
    # if episode % n_update == 0:
    #     model.target_update()
    # Discretize state into buckets
    # current_state, done = discretizer(*env.reset()), False

    while done == False:

        if np.random.random() < epsilon:
           action = env.action_space.sample()
        else:
            q_values = model.predict(state)
            action = torch.argmax(q_values).item() 

        # Take action and add reward to total
        next_state, reward, done, _ = env.step(action)
        # increment environment
        # obs, reward, done, _ = env.step(action)
        q_values = model.predict(state).tolist()
        # model.update(state, q_values)
        q_values_next = model.predict(next_state)
        q_values[action] = reward + gamma * torch.max(q_values_next).item()
        model.update(state, q_values)
        state = next_state

        env.render()
    epsilon = max(epsilon * eps_decay, 0.01)
    print(e)



