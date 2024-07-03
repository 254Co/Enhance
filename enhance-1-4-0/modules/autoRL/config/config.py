# config/config.py

class Config:
    SPARK_APP_NAME = "RL Data Preparation"
    ENV_NAME = "CartPole-v1"
    TOTAL_TIMESTEPS = 10000
    MODEL_PATH = "ppo_cartpole"
    RAW_DATA_PATH = "path/to/raw_data.csv"
    NEW_DATA_PATH = "path/to/new_data.csv"
