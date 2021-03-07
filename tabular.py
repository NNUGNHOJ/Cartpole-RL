import gym
import time, math, random
import numpy as np
import sklearn
from sklearn.preprocessing import KBinsDiscretizer
from typing import Tuple

env = gym.make('CartPole-v0')

"""Discretize the environment by turning the CartPole continuous space
into a single discrete space."""

n_bins = (6, 12)
lower_bounds = [env.observation_space.low[2], -math.radians(50)]
upper_bounds = [env.observation_space.high[2], -math.radians(50)]


def discretizer(_, __, angle, pole_velocity) -> Tuple[int,...]:
    est = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="uniform")
    est.fit([lower_bounds, upper_bounds])
    return tuple(map(int, est.transform([[angle, pole_velocity]])[0]))


"""Initialize Q-table with zeros"""
Q_table = np.zeros(n_bins + (env.action_space.n,))


def policy(state: tuple):
    """Create policy function using the Q-table, greedily select highest Q value"""
    return np.argmax(Q_table[state])


def new_Q_value(reward: float, new_state: tuple, discount_factor=1) -> float:
    """Update Q-value of state/action pair"""
    future_optimal_value = np.max(Q_table[new_state])
    learned_value = reward + discount_factor * future_optimal_value
    return learned_value


def learning_rate(n: int, min_rate=0.01) -> float:
    """Decaying learning rate"""
    return max(min_rate, min(1.0, 1.0 - math.log10((n + 1) / 25)))


def exploration_rate(n: int, min_rate=0.1) -> float:
    """Decaying exploration rate"""
    return max(min_rate, min(1, 1.0 - math.log10((n + 1) / 25)))


"""Training..."""
n_episodes = 10000
for e in range(n_episodes):

    # Discretize state into buckets
    current_state, done = discretizer(*env.reset()), False

    while done == False:

        # policy action
        action = policy(current_state)  # exploit

        # insert random action
        if np.random.random() < exploration_rate(e):
            action = env.action_space.sample()  # explore

        # increment enviroment
        obs, reward, done, _ = env.step(action)
        new_state = discretizer(*obs)

        # Update Q-Table
        lr = learning_rate(e)
        learnt_value = new_Q_value(reward, new_state)
        old_value = Q_table[current_state][action]
        Q_table[current_state][action] = (1 - lr) * old_value + lr * learnt_value

        current_state = new_state

        # Render the cartpole environment
        env.render()