# train.py
from pyspark.sql import SparkSession
from enhance.Enhance.utils.logger import get_logger
from enhance.Enhance.utils.exception_handler import handle_exceptions
from .utils import preprocess_data
from .autoencoder import Autoencoder
from .polygon_websocket import start_polygon_stream
import numpy as np

logger = get_logger(__name__)

@handle_exceptions
def train_autoencoder(input_dim, bucket_name, model_name, polygon_api_key, batch_size=256):
    spark = SparkSession.builder.appName('AutoencoderTraining').getOrCreate()
    logger.info("Spark session started for Autoencoder training")

    autoencoder = Autoencoder(input_dim)
    logger.info(f"Autoencoder initialized with input dimension: {input_dim}")

    def process_batch(data):
        try:
            data = np.array(data)
            data = preprocess_data(data)
            autoencoder.train_batch(data, batch_size=batch_size)
            autoencoder.save_model(bucket_name, model_name)
            logger.info("Batch processed and model saved")
        except Exception as e:
            log_error(logger, e)
            raise e

    start_polygon_stream(polygon_api_key, process_batch)

    spark.stop()
    logger.info("Spark session stopped")
    return autoencoder