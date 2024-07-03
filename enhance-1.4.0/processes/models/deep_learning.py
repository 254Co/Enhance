from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import DataFrame
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_deep_learning(df: DataFrame, features: list, label: str):
    logger.info("Running deep learning model")
    try:
        assembler = VectorAssembler(inputCols=features, outputCol='features')
        df_prepared = assembler.transform(df)

        layers = [len(features), 128, 64, 2]  # Example architecture
        mlp = MultilayerPerceptronClassifier(layers=layers, featuresCol='features', labelCol=label)
        mlp_model = mlp.fit(df_prepared)

        logger.info("Deep learning model training completed successfully")
        return mlp_model
    except Exception as e:
        logger.error(f"Error during deep learning model training: {str(e)}", exc_info=True)
        raise e