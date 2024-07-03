"""
Model evaluation module.

Author: 254StudioZ LLC
Date: 2024-07-01
"""

from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import DataFrame

def evaluate_model(predictions: DataFrame) -> dict:
    """
    Evaluates the given model predictions using multiple metrics.

    Parameters:
    predictions (DataFrame): The DataFrame containing model predictions.

    Returns:
    dict: Dictionary containing evaluation metrics (accuracy, precision, recall, f1).
    """
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")

    accuracy = evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"})
    precision = evaluator.evaluate(predictions, {evaluator.metricName: "weightedPrecision"})
    recall = evaluator.evaluate(predictions, {evaluator.metricName: "weightedRecall"})
    f1 = evaluator.evaluate(predictions, {evaluator.metricName: "f1"})

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
