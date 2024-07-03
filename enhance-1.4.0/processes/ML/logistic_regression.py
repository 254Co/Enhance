# binary_classifier/logistic_regression.py
from pyspark.sql import DataFrame
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import logging

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def train_logistic_regression(input_df: DataFrame, label_col: str, feature_cols: list, train_test_split: tuple, seed: int):
    """
    Trains a logistic regression model on the input DataFrame.

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
        training_data, test_data = input_df.randomSplit(train_test_split, seed=seed)

        logger.info("Initializing LogisticRegression")
        logistic_regression = LogisticRegression(labelCol=label_col, featuresCol=feature_cols)

        logger.info("Creating pipeline with LogisticRegression")
        pipeline = Pipeline(stages=[logistic_regression])

        logger.info("Building parameter grid for cross-validation")
        param_grid = ParamGridBuilder() \
            .addGrid(logistic_regression.regParam, [0.01, 0.1, 1.0]) \
            .addGrid(logistic_regression.maxIter, [10, 50, 100]) \
            .build()

        logger.info("Setting up cross-validator")
        cross_validator = CrossValidator(
            estimator=pipeline,
            estimatorParamMaps=param_grid,
            evaluator=BinaryClassificationEvaluator(),
            numFolds=3,
            parallelism=4  # Utilize parallelism for cross-validation
        )

        logger.info("Fitting cross-validator to training data")
        cv_model = cross_validator.fit(training_data)

        logger.info("Model training completed successfully")
        return cv_model, test_data
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}", exc_info=True)
        raise e