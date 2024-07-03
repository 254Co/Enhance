# gpt_integration/gpt_utils.py

import openai

def define_problem_with_gpt(description):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"Define the reinforcement learning problem based on the following description: {description}",
        max_tokens=150
    )
    return response.choices[0].text.strip()

def feature_engineering_with_gpt(raw_data_description):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"Suggest feature engineering techniques for the following raw data: {raw_data_description}",
        max_tokens=150
    )
    return response.choices[0].text.strip()

def recommend_algorithm_with_gpt(problem_description):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"Recommend an RL algorithm for the following problem: {problem_description}",
        max_tokens=50
    )
    return response.choices[0].text.strip()

def suggest_hyperparameters_with_gpt(algorithm, env_name):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"Suggest initial hyperparameters for training a {algorithm} agent in the {env_name} environment.",
        max_tokens=150
    )
    return response.choices[0].text.strip()

def analyze_metrics_with_gpt(metrics):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"Analyze the following performance metrics and suggest improvements: {metrics}",
        max_tokens=150
    )
    return response.choices[0].text.strip()

def generate_deployment_script_with_gpt(model_name):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"Generate a Python script to deploy the trained model {model_name}.",
        max_tokens=150
    )
    return response.choices[0].text.strip()
