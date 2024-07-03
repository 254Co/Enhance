# data_loader.py
from pyspark.sql import SparkSession, DataFrame

def load_data(spark: SparkSession, file_path: str) -> DataFrame:
    """
    Load market data from a specified file path.
    
    Parameters:
    - spark: SparkSession object.
    - file_path: Path to the CSV file containing market data.
    
    Returns:
    - DataFrame: Spark DataFrame containing the market data.
    """
    return spark.read.csv(file_path, header=True, inferSchema=True)
