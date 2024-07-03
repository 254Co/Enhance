# binary_classifier/naive_bayes.py
from pyspark.sql import DataFrame
from pyspark.ml.classification import NaiveBayes
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import logging

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def train_naive_bayes(input_df: DataFrame, label_col: str, feature_cols: list, train_test_split: tuple, seed: int):
    """
    Trains a naive Bayes model on the input DataFrame.

    Args:
        input_df (DataFrame): The input Spark DataFrame.
        label_col (str): The name of the label column.
        feature_cols (list): List of feature column names.
        train_test_split (tuple): Tuple representing the train-test split ratio.
        seed (int): Random seed for splitting data.

    Returns:
        Tuple[PipelineModel, DataFrame]: The trained model and the test DataFrame.
    """
    try:
        logger.info("Splitting data into training and test sets")
        trainingData, testData = input_df.randomSplit(train_test_split, seed=seed)

        logger.info("Initializing NaiveBayes")
        nb = NaiveBayes(labelCol=label_col, featuresCol=feature_cols)

        logger.info("Creating pipeline with NaiveBayes")
        pipeline = Pipeline(stages=[nb])

        logger.info("Building parameter grid for cross-validation")
        paramGrid = ParamGridBuilder() \
            .addGrid(nb.smoothing, [0.0, 0.5, 1.0]) \
            .build()

        logger.info("Setting up cross-validator")
        crossval = CrossValidator(
            estimator=pipeline,
            estimatorParamMaps=paramGrid,
            evaluator=BinaryClassificationEvaluator(),
            numFolds=3,
            parallelism=4  # Utilize parallelism for cross-validation
        )

        logger.info("Fitting cross-validator to training data")
        cvModel = crossval.fit(trainingData)

        logger.info("Model training completed successfully")
        return cvModel, testData
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}", exc_info=True)
        raise e