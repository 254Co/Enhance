"""
Logistic Regression training and tuning module using Spark MLlib.

Author: 254StudioZ LLC
Date: 2024-07-01
"""

from pyspark.ml.classification import LogisticRegression, LogisticRegressionModel
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import DataFrame

def train_logistic_regression(train_data: DataFrame) -> LogisticRegressionModel:
    """
    Trains a Logistic Regression model on the given training data.

    Parameters:
    train_data (DataFrame): The training data.

    Returns:
    LogisticRegressionModel: The trained Logistic Regression model.
    """
    lr = LogisticRegression(featuresCol='scaledFeatures', labelCol='label')
    model = lr.fit(train_data)
    return model

def tune_logistic_regression(train_data: DataFrame) -> (LogisticRegressionModel, dict):
    """
    Tunes a Logistic Regression model using CrossValidator to find the best hyperparameters.

    Parameters:
    train_data (DataFrame): The training data.

    Returns:
    (LogisticRegressionModel, dict): The best model and its hyperparameters.
    """
    lr = LogisticRegression(featuresCol='scaledFeatures', labelCol='label')
    paramGrid = (ParamGridBuilder()
                 .addGrid(lr.regParam, [0.01, 0.1, 1.0])
                 .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])
                 .build())

    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")

    cv = CrossValidator(estimator=lr, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=3)
    cv_model = cv.fit(train_data)

    best_model = cv_model.bestModel
    best_params = {param.name: best_model.getOrDefault(param) for param in paramGrid[0]}
    
    return best_model, best_params
