# intelligence.py
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import avg, stddev

def extract_intelligence(data: DataFrame, spark: SparkSession) -> DataFrame:
    """
    Extract actionable intelligence from the fractal analysis results.
    
    Parameters:
    - data: Input Spark DataFrame containing market data and calculated fractal features.
    - spark: SparkSession object.
    
    Returns:
    - DataFrame: Spark DataFrame containing intelligence insights.
    """
    
    # Calculate summary statistics for fractal dimensions
    summary_stats = data.select(
        [avg(col).alias(f"avg_{col}") for col in data.columns if col.startswith("fractal_")] +
        [stddev(col).alias(f"stddev_{col}") for col in data.columns if col.startswith("fractal_")]
    )
    
    # Extract key intelligence insights
    insights = summary_stats.withColumn("market_stability", col("avg_fractal_dimension_30"))
    insights = insights.withColumn("market_volatility", col("stddev_fractal_dimension_30"))
    
    return insights

def save_intelligence(insights: DataFrame, path: str):
    """
    Save the extracted intelligence insights to a specified path.
    
    Parameters:
    - insights: Spark DataFrame containing intelligence insights.
    - path: Path to save the insights.
    """
    insights.write.csv(path, header=True)
