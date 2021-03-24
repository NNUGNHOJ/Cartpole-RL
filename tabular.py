import gym
import time, math, random
import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import KBinsDiscretizer
from typing import Tuple


def discretizer(position, __, angle, pole_velocity, lower_bounds, upper_bounds, n_bins):
    est = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="uniform")
    est.fit([lower_bounds, upper_bounds])
    return tuple(map(int, est.transform([[angle, (pole_velocity)]])[0]))


def policy(state: tuple, Q_table):
    """Create policy function using the Q-table, greedily select highest Q value"""
    return np.argmax(Q_table[state])


def new_Q_value(reward, new_state, Q_table, discount_factor=1):
    """Update Q-value of state/action pair"""
    future_optimal_value = np.max(Q_table[new_state])
    learned_value = reward + discount_factor * future_optimal_value
    return learned_value


def learning_rate(n: int, min_rate=0.01):
    """Decaying learning rate"""
    return max(min_rate, min(1.0, 1.0 - math.log10((n + 1) / 25)))


def exploration_rate(n: int, min_rate=0.001):
    """Decaying exploration rate"""
    return max(min_rate, min(1, 1.0 - math.log10((n + 1) / 25)))


def train_default():
    """Training..."""
    env = gym.make('CartPole-v1')
    lower_bounds = [env.observation_space.low[2] / 3, -1]
    upper_bounds = [env.observation_space.high[2] / 3, 1]
    n_bins = (24, 4)

    n_episodes = 10000

    """Initialize Q-table with zeros"""
    # TODO: experiment with filling with different values not just zeros.
    Q_table = np.zeros(n_bins + (env.action_space.n,))

    for e in range(n_episodes):

        total=0

        # Discretize state into buckets
        current_state, done = discretizer(*env.reset(), lower_bounds, upper_bounds, n_bins), False

        while done == False:

            # policy action
            action = policy(current_state, Q_table)  # exploit
            # insert random action
            if np.random.random() < exploration_rate(e):
                action = env.action_space.sample()  # explore

            # increment environment
            obs, reward, done, _ = env.step(action)
            new_state = discretizer(*obs, lower_bounds, upper_bounds, n_bins)
            total += reward # experiment
            reward =  1/abs(obs[3]) * 1/abs(obs[2]) # why do we do the reward like this?

            # Update Q-Table
            lr = learning_rate(e)
            learnt_value = new_Q_value(reward, new_state, Q_table)
            old_value = Q_table[current_state][action]
            Q_table[current_state][action] = (1 - lr) * old_value + lr * learnt_value
            current_state = new_state

            # Render the cartpole environment
            env.render()
        print(total)
    return


def train_specific(n_bins, n_episodes, df):
    """Training with set bin size..."""
    env = gym.make('CartPole-v1')
    lower_bounds = [env.observation_space.low[2] / 3, -1]
    upper_bounds = [env.observation_space.high[2] / 3, 1]

    """Initialize Q-table with zeros"""
    # TODO: experiment with filling with different values not just zeros.
    Q_table = np.zeros(n_bins + (env.action_space.n,))

    for e in range(n_episodes):
        total = 0

        # Discretize state into buckets
        current_state, done = discretizer(*env.reset(), lower_bounds, upper_bounds, n_bins), False

        while done == False:

            # policy action
            action = policy(current_state, Q_table)  # exploit
            # insert random action
            if np.random.random() < exploration_rate(e):
                action = env.action_space.sample()  # explore

            # increment environment
            obs, reward, done, _ = env.step(action)
            new_state = discretizer(*obs, lower_bounds, upper_bounds, n_bins)
            total += reward  # experiment
            reward = 1 / abs(obs[3]) * 1 / abs(obs[2])  # why do we do the reward like this?

            # Update Q-Table
            lr = learning_rate(e)
            learnt_value = new_Q_value(reward, new_state, Q_table)
            old_value = Q_table[current_state][action]
            Q_table[current_state][action] = (1 - lr) * old_value + lr * learnt_value
            current_state = new_state

            # Render the cartpole environment
            #env.render()

        df.at[e, str(n_bins[0])] = total
        print(df)
        print(total)


    return df


def run_experiment():
    n_episodes = 1000

    df = pd.DataFrame(index=range(n_episodes), columns=['3', '6', '12', '24', '48', '96', '192'])
    bin_sizes = [3, 6, 12, 24, 48, 96, 192]

    print(df.shape)

    for bin_size in bin_sizes:
        train_specific((bin_size, 4), n_episodes, df)

    df.to_pickle('tabular_bins_experiment')
    return


def run_experiment_2():
    n_episodes = 1000

    df = pd.DataFrame(index=range(n_episodes), columns=['12'])
    bin_sizes = [12]

    print(df.shape)

    for bin_size in bin_sizes:
        train_specific((bin_size, 4), n_episodes, df)

    df.to_pickle('tabular_bins_experiment_2')
    return


run_experiment_2()


