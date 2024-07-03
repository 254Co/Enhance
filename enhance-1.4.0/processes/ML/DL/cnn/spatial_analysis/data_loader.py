# spatial_analysis/data_loader.py

from pyspark.sql import SparkSession

class DataLoader:
    def __init__(self):
        self.spark = SparkSession.builder.appName("SpatialDataAnalysisWithCNN").getOrCreate()

    def load_images(self, path):
        schema = StructType([
            StructField("path", StringType(), True),
            StructField("image", BinaryType(), True)
        ])
        images_df = self.spark.read.format("binaryFile").schema(schema).load(path)
        return images_df

    def load_labels(self, path):
        labels_df = self.spark.read.csv(path, header=True)
        return labels_df
