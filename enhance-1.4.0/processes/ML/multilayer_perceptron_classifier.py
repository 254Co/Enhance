from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import logging

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def multilayer_perceptron_classifier_process(input_df: DataFrame, label_col: str, feature_cols: list, layers: list):
    """
    Trains a multilayer perceptron classifier model on the input DataFrame.

    Args:
        input_df (DataFrame): The input Spark DataFrame.
        label_col (str): The name of the label column.
        feature_cols (list): List of feature column names.
        layers (list): List representing the number of nodes in each layer of the neural network.

    Returns:
        Tuple[PipelineModel, float]: The trained model and the AUC value.
    """
    try:
        logger.info("Indexing label column")
        indexer = StringIndexer(inputCol=label_col, outputCol="indexedLabel")

        logger.info("Assembling feature columns into a feature vector")
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

        logger.info("Creating MultilayerPerceptronClassifier model")
        mlp = MultilayerPerceptronClassifier(labelCol="indexedLabel", featuresCol="features", layers=layers)

        logger.info("Creating pipeline with indexer, assembler, and MultilayerPerceptronClassifier")
        pipeline = Pipeline(stages=[indexer, assembler, mlp])

        logger.info("Splitting data into training and test sets")
        trainingData, testData = input_df.randomSplit([0.8, 0.2], seed=1234)

        logger.info("Building parameter grid for cross-validation")
        paramGrid = ParamGridBuilder() \
            .addGrid(mlp.maxIter, [100, 200]) \
            .addGrid(mlp.stepSize, [0.01, 0.1]) \
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