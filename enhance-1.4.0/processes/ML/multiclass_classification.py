from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
import logging
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def multiclass_classifier_process(input_df: DataFrame, label_col: str, feature_cols: list):
    try:
        logger.info("Indexing label column")
        indexer = StringIndexer(inputCol=label_col, outputCol="indexedLabel")

        logger.info("Assembling feature columns into a feature vector")
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

        logger.info("Creating RandomForestClassifier model")
        rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="features")

        logger.info("Creating pipeline with indexer, assembler, and RandomForestClassifier")
        pipeline = Pipeline(stages=[indexer, assembler, rf])

        logger.info("Splitting data into training and test sets")
        trainingData, testData = input_df.randomSplit([0.8, 0.2], seed=1234)

        logger.info("Building parameter grid for cross-validation")
        paramGrid = ParamGridBuilder() \
            .addGrid(rf.numTrees, [20, 50, 100]) \
            .addGrid(rf.maxDepth, [5, 10, 15]) \
            .build()

        logger.info("Setting up cross-validator")
        crossval = CrossValidator(
            estimator=pipeline,
            estimatorParamMaps=paramGrid,
            evaluator=MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy"),
            numFolds=3,
            parallelism=4  # Utilize parallelism for cross-validation
        )

        logger.info("Fitting cross-validator to training data")
        cvModel = crossval.fit(trainingData)

        logger.info("Making predictions on the test data")
        predictions = cvModel.transform(testData)

        logger.info("Evaluating the model")
        evaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
        accuracy = evaluator.evaluate(predictions)

        logger.info(f"Accuracy = {accuracy}")

        return cvModel, accuracy
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}", exc_info=True)
        raise e