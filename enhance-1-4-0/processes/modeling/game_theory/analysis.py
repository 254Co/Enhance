import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_game_outcomes(game_results: np.ndarray) -> np.ndarray:
    logger.info("Analyzing game outcomes")
    strategies = np.argmax(game_results, axis=1)
    frequencies = np.bincount(strategies) / len(strategies)
    return frequencies
