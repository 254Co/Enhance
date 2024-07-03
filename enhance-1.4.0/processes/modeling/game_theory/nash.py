import numpy as np
from scipy.optimize import linprog
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compute_nash_equilibrium(payoff_matrix: np.ndarray) -> np.ndarray:
    logger.info("Computing Nash equilibrium")
    num_strategies = payoff_matrix.shape[0]
    c = [-1] + [0] * num_strategies
    A_ub = np.hstack([np.ones((num_strategies, 1)), -payoff_matrix])
    b_ub = [0] * num_strategies
    A_eq = [0] + [1] * num_strategies
    b_eq = [1]

    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=[A_eq], b_eq=b_eq, bounds=(None, None))
        if result.success:
            logger.info("Nash equilibrium computed successfully")
            return result.x[1:]
        else:
            logger.error(f"Failed to compute Nash equilibrium: {result.message}")
            raise ValueError("Failed to compute Nash equilibrium")
    except Exception as e:
        logger.error(f"Error computing Nash equilibrium: {str(e)}", exc_info=True)
        raise e

def compute_mixed_strategy_nash(payoff_matrix: np.ndarray) -> np.ndarray:
    logger.info("Computing mixed strategy Nash equilibrium")
    num_strategies = payoff_matrix.shape[0]
    c = [-1] + [0] * num_strategies
    A_ub = np.hstack([np.ones((num_strategies, 1)), -payoff_matrix])
    b_ub = [0] * num_strategies
    A_eq = [[0] + [1] * num_strategies]
    b_eq = [1]

    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=(None, None))
        if result.success:
            logger.info("Mixed strategy Nash equilibrium computed successfully")
            return result.x[1:]
        else:
            logger.error(f"Failed to compute mixed strategy Nash equilibrium: {result.message}")
            raise ValueError("Failed to compute mixed strategy Nash equilibrium")
    except Exception as e:
        logger.error(f"Error computing mixed strategy Nash equilibrium: {str(e)}", exc_info=True)
        raise e