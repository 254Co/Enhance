from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from pyspark.sql import DataFrame
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_transfer_learning(df: DataFrame, features: list, label: str, base_model):
    logger.info("Running transfer learning")
    base_model.trainable = False
    x = base_model.output
    predictions = Dense(1, activation='linear')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer='adam', loss='mean_squared_error')

    X = df.select(features).toPandas().values
    y = df.select(label).toPandas().values
    model.fit(X, y, epochs=10, batch_size=32)

    return model