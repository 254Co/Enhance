import os
import yaml
import logging

# Initialize logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def load_config(config_file='config/config.yaml'):
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        logger.info("Configuration file loaded successfully")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration file: {str(e)}", exc_info=True)
        raise

config = load_config()

# Load environment variables
try:
    config['openai']['api_key'] = os.getenv('OPENAI_API_KEY', config['openai']['api_key'])
    config['database']['password'] = os.getenv('DB_PASSWORD', config['database']['password'])
    logger.info("Environment variables loaded successfully")
except Exception as e:
    logger.error(f"Error loading environment variables: {str(e)}", exc_info=True)
    raise