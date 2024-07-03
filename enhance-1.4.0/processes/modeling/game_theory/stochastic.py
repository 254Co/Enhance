import numpy as np
from scipy.optimize import linprog
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compute_stochastic_nash_equilibrium(payoff_matrices):
    logger.info("Computing stochastic Nash equilibrium")
    num_states = len(payoff_matrices)
    equilibria = []

    for state in range(num_states):
        payoff_matrix = payoff_matrices[state]
        num_strategies = payoff_matrix.shape[0]
        c = [-1] + [0] * num_strategies
        A_ub = np.hstack([np.ones((num_strategies, 1)), -payoff_matrix])
        b_ub = [0] * num_strategies
        A_eq = [[0] + [1] * num_strategies]
        b_eq = [1]

        try:
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=(None, None))
            if result.success:
                logger.info(f"Stochastic Nash equilibrium for state {state} computed successfully")
                equilibria.append(result.x[1:])
            else:
                logger.error(f"Failed to compute stochastic Nash equilibrium for state {state}: {result.message}")
                raise ValueError(f"Failed to compute stochastic Nash equilibrium for state {state}")
        except Exception as e:
            logger.error(f"Error computing stochastic Nash equilibrium for state {state}: {str(e)}", exc_info=True)
            raise e

    return equilibria