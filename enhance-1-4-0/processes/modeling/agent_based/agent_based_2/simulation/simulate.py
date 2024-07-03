from pyspark.sql import SparkSession
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import *
from enhance.Enhance.utils.logger import get_logger
from enhance.Enhance.utils.exception_handler import handle_exceptions
from enhance.Enhance.data_processing.data_transformation.utils import ParallelProcessingUtils

logger = get_logger(__name__)

def run_simulation(agents, interactions, steps):
    """
    Run an agent-based simulation.

    Parameters:
    agents (list): List of agent objects.
    interactions (InteractionModel): Interaction model object.
    steps (int): Number of simulation steps.

    Returns:
    list: List of agents' states after the simulation.
    """
    try:
        logger.info(f"Starting simulation with {len(agents)} agents for {steps} steps.")
        for step in range(steps):
            logger.debug(f"Simulation step {step + 1}/{steps} started.")
            interactions.interact()
            for agent in agents:
                agent.step()
            logger.debug(f"Simulation step {step + 1}/{steps} completed.")
        final_states = [agent.get_state() for agent in agents]
        logger.info("Simulation completed successfully.")
        return final_states
    except Exception as e:
        logger.error(f"Error during simulation: {str(e)}", exc_info=True)
        raise

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