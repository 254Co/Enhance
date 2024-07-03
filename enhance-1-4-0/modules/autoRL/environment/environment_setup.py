# environment/environment_setup.py

import gym

def setup_environment(env_name):
    env = gym.make(env_name)
    return env
