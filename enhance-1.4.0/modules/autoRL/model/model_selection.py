# model/model_selection.py

from stable_baselines3 import PPO

def select_model(env, hyperparameters):
    model = PPO('MlpPolicy', env, verbose=1, **hyperparameters)
    return model
