from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import logging

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def decision_tree_classifier_process(input_df: DataFrame, label_col: str, feature_cols: list):
    try:
        logger.info("Indexing label column")
        indexer = StringIndexer(inputCol=label_col, outputCol="indexedLabel")

        logger.info("Assembling feature columns into a feature vector")
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

        logger.info("Creating DecisionTreeClassifier model")
        dt = DecisionTreeClassifier(labelCol="indexedLabel", featuresCol="features")

        logger.info("Creating pipeline with indexer, assembler, and DecisionTreeClassifier")
        pipeline = Pipeline(stages=[indexer, assembler, dt])

        logger.info("Splitting data into training and test sets")
        trainingData, testData = input_df.randomSplit([0.8, 0.2], seed=1234)

        logger.info("Training the model")
        model = pipeline.fit(trainingData)

        logger.info("Making predictions on the test data")
        predictions = model.transform(testData)

        logger.info("Evaluating the model")
        evaluator = BinaryClassificationEvaluator(labelCol="indexedLabel", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
        auc = evaluator.evaluate(predictions)

        logger.info(f"Area Under ROC = {auc}")

        return model, auc
    except Exception as e:
        logger.error(f"Error in decision tree classifier process: {str(e)}", exc_info=True)
        raise e