from pyspark.ml.classification import RandomForestClassifier, GBTClassifier, LogisticRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import DataFrame
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_random_forest(df: DataFrame, features: list, label: str) -> RandomForestClassifier:
    logger.info("Running Random Forest classifier")
    assembler = VectorAssembler(inputCols=features, outputCol='features')
    df_prepared = assembler.transform(df)

    rf = RandomForestClassifier(featuresCol='features', labelCol=label)
    rf_model = rf.fit(df_prepared)

    return rf_model

def run_gbt(df: DataFrame, features: list, label: str) -> GBTClassifier:
    logger.info("Running GBT classifier")
    assembler = VectorAssembler(inputCols=features, outputCol='features')
    df_prepared = assembler.transform(df)

    gbt = GBTClassifier(featuresCol='features', labelCol=label)
    gbt_model = gbt.fit(df_prepared)

    return gbt_model

def run_logistic_regression(df: DataFrame, features: list, label: str) -> LogisticRegression:
    logger.info("Running Logistic Regression")
    assembler = VectorAssembler(inputCols=features, outputCol='features')
    df_prepared = assembler.transform(df)

    lr = LogisticRegression(featuresCol='features', labelCol=label)
    lr_model = lr.fit(df_prepared)

    return lr_model

def analyze_classification_results(model, test_data: DataFrame) -> float:
    logger.info("Analyzing classification results")
    predictions = model.transform(test_data)
    evaluator = MulticlassClassificationEvaluator(labelCol='label', predictionCol='prediction', metricName='accuracy')
    accuracy = evaluator.evaluate(predictions)
    return accuracy

def feature_importance(model) -> list:
    logger.info("Getting feature importance")
    return model.featureImportances