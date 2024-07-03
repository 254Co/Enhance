# time_series_analysis/inference.py

import numpy as np
from pyspark.sql.functions import pandas_udf, PandasUDFType

class Predictor:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)

    def predict(self, sequences):
        sequences = np.array(sequences.tolist())
        predictions = self.model.predict(sequences)
        return predictions.tolist()

    def predict_udf(self):
        return pandas_udf(self.predict, "array<float>", PandasUDFType.SCALAR)

    def predict_sequences(self, processed_data_df):
        predict_udf = self.predict_udf()
        predictions_df = processed_data_df.withColumn("predictions", predict_udf(col("sequences")))
        return predictions_df
