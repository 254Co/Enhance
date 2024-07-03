# copula_analysis/data_preparation.py
from pyspark.sql import SparkSession

def prepare_data(filepath):
    spark = SparkSession.builder.appName("CopulaAnalysis").getOrCreate()
    data = spark.read.csv(filepath, header=True, inferSchema=True)
    data = data.dropna()
    return data
