import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from a .env file if present

# Ensure required environment variables are set
required_env_vars = ['OPENAI_API_KEY', 'DB_PASSWORD']

for var in required_env_vars:
    if var not in os.environ:
        raise EnvironmentError(f"Required environment variable '{var}' not set.")