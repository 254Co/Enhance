# scripts/deploy_agent.py

from auto_rl.config.config import Config
from auto_rl.deployment.deployment import deploy_model
from stable_baselines3 import PPO

def main():
    # Load the trained model
    model = PPO.load(Config.MODEL_PATH)
    
    # Deploy the model
    deploy_model(model, Config.MODEL_PATH)

if __name__ == "__main__":
    main()
