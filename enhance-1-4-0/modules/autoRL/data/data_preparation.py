# data/data_preparation.py

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lag
from pyspark.sql.window import Window

def init_spark(app_name):
    spark = SparkSession.builder.appName(app_name).getOrCreate()
    return spark

def prepare_data(spark, raw_data_path):
    raw_data = spark.read.csv(raw_data_path, header=True, inferSchema=True)
    windowSpec = Window.orderBy("timestamp")
    processed_data = raw_data.withColumn("prev_cart_position", lag(col("cart_position")).over(windowSpec)) \
                             .withColumn("cart_position_change", col("cart_position") - col("prev_cart_position"))
    return processed_data
