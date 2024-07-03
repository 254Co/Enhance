from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
import logging

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def clustering_process(input_df: DataFrame, feature_cols: list, k: int):
    """
    Perform KMeans clustering on the input DataFrame.

    Args:
        input_df (DataFrame): The input Spark DataFrame.
        feature_cols (list): List of feature column names.
        k (int): Number of clusters.

    Returns:
        Tuple[PipelineModel, float]: The trained KMeans model and the silhouette score.
    """
    try:
        logger.info("Assembling feature columns into a feature vector")
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

        logger.info("Creating KMeans clustering model")
        kmeans = KMeans(k=k, featuresCol="features")

        logger.info("Creating pipeline with assembler and KMeans model")
        pipeline = Pipeline(stages=[assembler, kmeans])

        logger.info("Training the KMeans model")
        model = pipeline.fit(input_df)

        logger.info("Making predictions on the input data")
        predictions = model.transform(input_df)

        logger.info("Evaluating the model using silhouette score")
        evaluator = ClusteringEvaluator(featuresCol="features", predictionCol="prediction", metricName="silhouette")
        silhouette = evaluator.evaluate(predictions)

        logger.info(f"Silhouette Score = {silhouette}")

        return model, silhouette
    except Exception as e:
        logger.error(f"Error during clustering process: {str(e)}", exc_info=True)
        raise e