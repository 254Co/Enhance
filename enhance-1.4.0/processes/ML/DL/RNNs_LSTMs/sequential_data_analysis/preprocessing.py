# sequential_data_analysis/preprocessing.py

import numpy as np
from pyspark.sql.functions import col, udf
from pyspark.sql.types import ArrayType, FloatType

class Preprocessor:
    def __init__(self, config):
        self.sequence_length = config.SEQUENCE_LENGTH

    def create_sequences(self, values):
        sequences = []
        for i in range(len(values) - self.sequence_length):
            seq = values[i:i + self.sequence_length]
            label = values[i + self.sequence_length]
            sequences.append((seq, label))
        return sequences

    def preprocess_udf(self):
        return udf(lambda values: self.create_sequences(values), ArrayType(ArrayType(FloatType())))

    def preprocess_data(self, data_df, column_name):
        preprocess_udf = self.preprocess_udf()
        processed_data_df = data_df.withColumn("sequences", preprocess_udf(col(column_name)))
        return processed_data_df
