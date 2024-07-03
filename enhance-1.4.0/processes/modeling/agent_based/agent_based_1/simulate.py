import numpy as np
import networkx as nx
from .agent import Agent
from pyspark.sql import SparkSession
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from enhance.Enhance.utils.logger import get_logger
from enhance.Enhance.utils.exception_handler import handle_exceptions
from enhance.Enhance.data_processing.data_transformation.utils import ParallelProcessingUtils

logger = get_logger(__name__)

def simulate_agents(num_agents: int, num_steps: int, initial_conditions=None, interaction_model=None, network=None, spatial=False):
    agents = initial_conditions if initial_conditions else [Agent(i, lambda x: np.random.choice([0, 1])) for i in range(num_agents)]
    if network is None:
        network = nx.complete_graph(num_agents)

    positions = None
    if spatial:
        positions = np.random.rand(num_agents, 2)  # Random initial positions

    for step in range(num_steps):
        for agent in agents:
            context = [agents[neighbor].history[-1] if agents[neighbor].history else 0 for neighbor in network.neighbors(agent.id)]
            decision = agent.decide(context)
            agent.history.append(decision)
            if interaction_model:
                interaction_model(agent, agents, step)
            if spatial:
                positions[agent.id] += np.random.randn(2) * 0.01

    return agents, positions

def advanced_interaction_model(agent_states: np.ndarray) -> np.ndarray:
    mean_state = np.mean(agent_states)
    adjusted_states = agent_states + (mean_state - agent_states) * 0.1
    return adjusted_states

def liquidity_shock_model(agent_states: np.ndarray) -> np.ndarray:
    shock = np.random.normal(0, 0.05, agent_states.shape)
    shocked_states = agent_states + shock
    return shocked_states

def analyze_agent_simulation(simulation_data: np.ndarray) -> tuple:
    avg_prices = np.mean(simulation_data, axis=0)
    volatility = np.std(simulation_data, axis=0)
    return avg_prices, volatility

def market_impact_model(agent_states: np.ndarray, impact_factor: float = 0.1) -> np.ndarray:
    impact = np.mean(agent_states) * impact_factor
    impacted_states = agent_states + impact
    return impacted_states

@handle_exceptions
def parallel_run_simulation(agents, interactions, steps, num_partitions: int = None):
    """
    Run an agent-based simulation with parallel processing.

    Parameters:
    agents (list): List of agent objects.
    interactions (InteractionModel): Interaction model object.
    steps (int): Number of simulation steps.
    num_partitions (int): Number of partitions for parallel processing.

    Returns:
    list: List of agents' states after the simulation.
    """
    try:
        logger.info(f"Starting parallel simulation with {len(agents)} agents for {steps} steps.")
        spark = SparkSession.builder.getOrCreate()
        agents_df = spark.createDataFrame(agents)

        def process_partition(iterator):
            local_agents = list(iterator)
            for step in range(steps):
                interactions.interact()
                for agent in local_agents:
                    agent.step()
            return [agent.get_state() for agent in local_agents]

        parallel_utils = ParallelProcessingUtils()
        result_df = parallel_utils.parallelize_dataframe_processing(agents_df, process_partition, num_partitions or spark.sparkContext.defaultParallelism)
        final_states = result_df.collect()
        logger.info("Parallel simulation completed successfully.")
        return final_states
    except Exception as e:
        logger.error(f"Error during parallel simulation: {str(e)}", exc_info=True)
        raise