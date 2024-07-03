# spatial_analysis/inference.py

import numpy as np
from pyspark.sql.functions import pandas_udf, PandasUDFType

class Predictor:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)

    def predict(self, images):
        images = np.array(images.tolist())
        predictions = self.model.predict(images)
        return predictions.tolist()

    def predict_udf(self):
        return pandas_udf(self.predict, "array<float>", PandasUDFType.SCALAR)

    def predict_images(self, new_processed_images_df):
        predict_udf = self.predict_udf()
        predictions_df = new_processed_images_df.withColumn("predictions", predict_udf(col("processed_image")))
        return predictions_df
