# model/training.py

def train_model(model, total_timesteps):
    model.learn(total_timesteps=total_timesteps)
    return model
