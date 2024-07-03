import logging
import sys

# Configure the logger globally for the entire application
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("application.log")
    ]
)

def get_logger(name):
    return logging.getLogger(name)

def log_error(logger, error):
    logger.error(f"Error: {str(error)}", exc_info=True)