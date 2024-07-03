import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def save_model(model, path: str):
    logger.info(f"Saving model to {path}")
    joblib.dump(model, path)

def load_model(path: str):
    logger.info(f"Loading model from {path}")
    return joblib.load(path)
