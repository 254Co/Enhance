from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import logging

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def regression_process(input_df: DataFrame, label_col: str, feature_cols: list):
    """
    Trains a linear regression model on the input DataFrame.

    Args:
        input_df (DataFrame): The input Spark DataFrame.
        label_col (str): The name of the label column.
        feature_cols (list): List of feature column names.

    Returns:
        Tuple[PipelineModel, float]: The trained model and the RMSE value.
    """
    try:
        logger.info("Assembling feature columns into a feature vector")
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

        logger.info("Creating LinearRegression model")
        lr = LinearRegression(labelCol=label_col, featuresCol="features")

        logger.info("Creating pipeline with assembler and LinearRegression")
        pipeline = Pipeline(stages=[assembler, lr])

        logger.info("Splitting data into training and test sets")
        trainingData, testData = input_df.randomSplit([0.8, 0.2], seed=1234)

        logger.info("Building parameter grid for cross-validation")
        paramGrid = ParamGridBuilder() \
            .addGrid(lr.regParam, [0.01, 0.1, 1.0]) \
            .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]) \
            .build()

        logger.info("Setting up cross-validator")
        crossval = CrossValidator(
            estimator=pipeline,
            estimatorParamMaps=paramGrid,
            evaluator=RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName="rmse"),
            numFolds=3,
            parallelism=4  # Utilize parallelism for cross-validation
        )

        logger.info("Fitting cross-validator to training data")
        cvModel = crossval.fit(trainingData)

        logger.info("Making predictions on the test data")
        predictions = cvModel.transform(testData)

        logger.info("Evaluating the model")
        evaluator = RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName="rmse")
        rmse = evaluator.evaluate(predictions)

        logger.info(f"Root Mean Squared Error = {rmse}")

        return cvModel, rmse
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}", exc_info=True)
        raise e