# time_series_analysis/data_loader.py

from pyspark.sql import SparkSession

class DataLoader:
    def __init__(self):
        self.spark = SparkSession.builder.appName("TimeSeriesAnalysisWithRNN").getOrCreate()

    def load_data(self, path):
        data_df = self.spark.read.csv(path, header=True, inferSchema=True)
        return data_df
