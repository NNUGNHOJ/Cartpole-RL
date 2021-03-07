import gym
import time

env = gym.make('CartPole-v0')

def sample_policy():
    """Uses a working policy from the sample library as an example.
    The values printed out are an array:
    [ cart_position, cart_velocity, pole_angle, pole_velocity_at_tip ]"""

    env = gym.make('CartPole-v0')
    for i_episode in range(20):
        observation = env.reset()
        for t in range(100):
            env.render()
            print(observation)
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
    env.close()


def constant_right_policy():
    """Runs 3 experiments where the cart is just told to go to the right at
    every state."""

    policy = lambda obs: 1

    for i_episode in range(3):
        obs = env.reset()
        for t in range(100):
            actions = policy(obs)
            obs, reward, done, info = env.step(actions)
            env.render()
            time.sleep(0.05)

    env.close()
    return

