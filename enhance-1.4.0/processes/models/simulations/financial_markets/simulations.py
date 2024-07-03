from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg
from pyspark.sql.types import DoubleType
from pyspark.sql import functions as F
import random
import concurrent.futures
import json
import logging
from processes.streaming import StreamingPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def simulate_financial_market(num_agents=1000, num_steps=100, num_threads=4,
                              initial_positions=None, historical_prices=None,
                              news_sentiment=None, external_factors=None):
    """
    Simulates a financial market using an agent-based model and returns the results as JSON.

    Args:
        num_agents (int): Number of agents in the simulation.
        num_steps (int): Number of simulation steps.
        num_threads (int): Number of threads for parallel simulation.
        initial_positions (list): Initial positions of agents.
        historical_prices (list): Historical prices for initializing the market.
        news_sentiment (dict): News sentiment data.
        external_factors (dict): External factor data.

    Returns:
        str: A JSON string containing the final market state and agent decisions.
    """
    # Create Spark session pointing to Spark master at localhost with local mode
    spark = SparkSession.builder \
        .appName("ABMMarket") \
        .master("local[*]")  # Use all available cores on the local machine
        .config("spark.executor.memory", "4g") \
        .config("spark.driver.memory", "4g") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "false") \
        .getOrCreate()

    logger.info("Spark session started for financial market simulation")

    # Create DataFrame of agents with additional attributes
    initial_positions = initial_positions or generate_initial_positions(num_agents)
    agents_data = [(i, random.choice(['fundamentalist', 'technical', 'noise']), initial_positions[i], random.uniform(1000, 10000)) for i in range(num_agents)]
    agents_df = spark.createDataFrame(agents_data, ["agent_id", "trader_type", "position", "capital"])

    # Broadcast agents_df
    agents_df.cache()

    # Initialize market state
    historical_prices = historical_prices or generate_historical_prices(num_steps)
    market_state = [(i, price) for i, price in enumerate(historical_prices)]
    market_df = spark.createDataFrame(market_state, ["time_step", "price"]).cache()

    # Function to simulate a single time step
    def simulate_step(time_step, market_df, agents_df):
        last_price = market_df.orderBy("time_step", ascending=False).first()["price"]

        # Get external factors and news sentiment if available
        sentiment = news_sentiment.get(time_step, 0) if news_sentiment else 0
        external = external_factors.get(time_step, {}) if external_factors else {}

        # Combine all external factors into a single impact value (simplified example)
        external_impact = sum(external.values()) if external else 0

        def agent_decision(trader_type, last_price, sentiment, external_impact):
            # Example decision making with sentiment and external factors
            if trader_type == 'fundamentalist':
                return random.uniform(-1, 1) + sentiment + external_impact  # Simplified example
            elif trader_type == 'technical':
                return random.uniform(-1, 1) + sentiment + external_impact  # Simplified example
            else:
                return random.uniform(-1, 1) + sentiment + external_impact  # Simplified example

        # Use regular UDF instead of Pandas UDF
        agent_decision_udf = F.udf(lambda trader_type: agent_decision(trader_type, last_price, sentiment, external_impact), DoubleType())

        # Agents make decisions
        decisions_df = agents_df.withColumn("decision", agent_decision_udf(col("trader_type")))

        # Aggregate decisions to determine market impact
        avg_decision = decisions_df.agg(avg(col("decision"))).collect()[0][0]

        # Update market price based on average decision
        new_price = last_price + avg_decision  # Simplified price update

        # Append new market state
        new_market_state = [(time_step, new_price)]
        new_market_df = spark.createDataFrame(new_market_state, ["time_step", "price"])
        market_df = market_df.union(new_market_df).cache()

        return market_df, decisions_df

    # Function to run simulation steps in parallel
    def run_simulation(start_step, end_step, market_df, agents_df):
        for step in range(start_step, end_step + 1):
            market_df, decisions_df = simulate_step(step, market_df, agents_df)
        return market_df, decisions_df

    # Additional simulations
    def simulate_volatility(market_df):
        """
        Simulates and calculates market volatility.
        """
        market_prices = market_df.select("price").collect()
        prices = [row["price"] for row in market_prices]
        mean_price = sum(prices) / len(prices)
        variance = sum((p - mean_price) ** 2 for p in prices) / len(prices)
        volatility = variance ** 0.5
        return volatility

    def simulate_market_shocks(market_df, shock_probability=0.05):
        """
        Simulates market shocks based on a given probability.
        """
        shock_impacts = []
        for row in market_df.collect():
            if random.random() < shock_probability:
                shock_impact = random.uniform(-10, 10)  # Example shock impact
                shock_impacts.append((row["time_step"], shock_impact))
            else:
                shock_impacts.append((row["time_step"], 0))
        return shock_impacts

    # Run simulation for a specified number of time steps using multithreading
    steps_per_thread = num_steps // num_threads

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for i in range(num_threads):
            start_step = i * steps_per_thread + 1
            end_step = (i + 1) * steps_per_thread
            futures.append(executor.submit(run_simulation, start_step, end_step, market_df, agents_df))

        results = [future.result() for future in concurrent.futures.as_completed(futures)]

    # Combine the results from all threads
    final_market_df = results[0][0]
    for df in results[1:]:
        final_market_df = final_market_df.union(df[0]).cache()

    final_decisions_df = results[0][1]
    for df in results[1:]:
        final_decisions_df = final_decisions_df.union(df[1]).cache()

    # Collect final results
    final_market_data = final_market_df.orderBy("time_step").collect()
    final_decisions_data = final_decisions_df.collect()

    # Perform additional simulations
    volatility = simulate_volatility(final_market_df)
    market_shocks = simulate_market_shocks(final_market_df)

    # Convert final results to JSON
    market_results = [{"time_step": row["time_step"], "price": row["price"]} for row in final_market_data]
    decisions_results = [{"agent_id": row["agent_id"], "trader_type": row["trader_type"], "decision": row["decision"], "position": row["position"], "capital": row["capital"]} for row in final_decisions_data]

    results_json = json.dumps({
        "market_results": market_results,
        "decisions_results": decisions_results,
        "volatility": volatility,
        "market_shocks": market_shocks
    })

    return results_json

def start_real_time_processing(input_kafka_topic: str, output_kafka_topic: str, kafka_bootstrap_servers: str):
    """
    Starts the real-time data processing using the StreamingPipeline.

    Args:
        input_kafka_topic (str): The name of the input Kafka topic.
        output_kafka_topic (str): The name of the output Kafka topic.
        kafka_bootstrap_servers (str): The Kafka bootstrap servers.
    """
    try:
        logger.info("Starting real-time data processing")
        StreamingPipeline.start_streaming(input_kafka_topic, output_kafka_topic, kafka_bootstrap_servers)
    except Exception as e:
        logger.error(f"Error in real-time data processing: {str(e)}", exc_info=True)
        raise