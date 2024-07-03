from pyspark.sql import DataFrame
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import logging

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def gbt_classifier_process(input_df: DataFrame, label_col: str, feature_cols: list):
    """
    Trains a gradient-boosted trees classifier model on the input DataFrame.

    Args:
        input_df (DataFrame): The input Spark DataFrame.
        label_col (str): The name of the label column.
        feature_cols (list): List of feature column names.

    Returns:
        Tuple[PipelineModel, float]: The trained model and the AUC value.
    """
    try:
        logger.info("Indexing label column")
        indexer = StringIndexer(inputCol=label_col, outputCol="indexedLabel")

        logger.info("Assembling feature columns into a feature vector")
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

        logger.info("Creating GBTClassifier model")
        gbt = GBTClassifier(labelCol="indexedLabel", featuresCol="features")

        logger.info("Creating pipeline with indexer, assembler, and GBTClassifier")
        pipeline = Pipeline(stages=[indexer, assembler, gbt])

        logger.info("Splitting data into training and test sets")
        trainingData, testData = input_df.randomSplit([0.8, 0.2], seed=1234)

        logger.info("Building parameter grid for cross-validation")
        paramGrid = ParamGridBuilder() \
            .addGrid(gbt.maxDepth, [5, 10, 15]) \
            .addGrid(gbt.maxIter, [10, 20, 50]) \
            .build()

        logger.info("Setting up cross-validator")
        crossval = CrossValidator(
            estimator=pipeline,
            estimatorParamMaps=paramGrid,
            evaluator=BinaryClassificationEvaluator(labelCol="indexedLabel", rawPredictionCol="rawPrediction", metricName="areaUnderROC"),
            numFolds=3,
            parallelism=4  # Utilize parallelism for cross-validation
        )

        logger.info("Fitting cross-validator to training data")
        cvModel = crossval.fit(trainingData)

        logger.info("Making predictions on the test data")
        predictions = cvModel.transform(testData)

        logger.info("Evaluating the model")
        evaluator = BinaryClassificationEvaluator(labelCol="indexedLabel", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
        auc = evaluator.evaluate(predictions)

        logger.info(f"Area Under ROC = {auc}")

        return cvModel, auc
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}", exc_info=True)
        raise e