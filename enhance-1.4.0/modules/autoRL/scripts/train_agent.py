# scripts/train_agent.py

from auto_rl.config.config import Config
from auto_rl.data.data_preparation import init_spark, prepare_data
from auto_rl.environment.environment_setup import setup_environment
from auto_rl.model.model_selection import select_model
from auto_rl.model.training import train_model
from auto_rl.model.evaluation import evaluate_model
from auto_rl.gpt_integration.gpt_utils import (
    define_problem_with_gpt,
    recommend_algorithm_with_gpt,
    suggest_hyperparameters_with_gpt
)

def main():
    # Define the problem with GPT
    problem_description = "Train an agent to balance a pole on a cart in the CartPole-v1 environment."
    problem_definition = define_problem_with_gpt(problem_description)
    print(problem_definition)
    
    # Initialize Spark
    spark = init_spark(Config.SPARK_APP_NAME)
    
    # Prepare data
    processed_data = prepare_data(spark, Config.RAW_DATA_PATH)
    
    # Setup environment
    env = setup_environment(Config.ENV_NAME)
    
    # Recommend algorithm with GPT
    rl_algorithm = recommend_algorithm_with_gpt(problem_description)
    print(f"Recommended algorithm: {rl_algorithm}")
    
    # Suggest hyperparameters with GPT
    hyperparameters = suggest_hyperparameters_with_gpt("PPO", Config.ENV_NAME)
    print(f"Suggested hyperparameters: {hyperparameters}")
    
    # Select model
    model = select_model(env, eval(hyperparameters))
    
    # Train model
    trained_model = train_model(model, Config.TOTAL_TIMESTEPS)
    
    # Evaluate model
    evaluate_model(env, trained_model)
    
    # Save the model
    trained_model.save(Config.MODEL_PATH)

if __name__ == "__main__":
    main()
