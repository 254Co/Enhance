"""
Random Forest training and tuning module using Spark MLlib.

Author: 254StudioZ LLC
Date: 2024-07-01
"""

from pyspark.ml.classification import RandomForestClassifier, RandomForestClassificationModel
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import DataFrame

def train_random_forest(train_data: DataFrame) -> RandomForestClassificationModel:
    """
    Trains a Random Forest model on the given training data.

    Parameters:
    train_data (DataFrame): The training data.

    Returns:
    RandomForestClassificationModel: The trained Random Forest model.
    """
    rf = RandomForestClassifier(featuresCol='scaledFeatures', labelCol='label')
    model = rf.fit(train_data)
    return model

def tune_random_forest(train_data: DataFrame) -> (RandomForestClassificationModel, dict):
    """
    Tunes a Random Forest model using CrossValidator to find the best hyperparameters.

    Parameters:
    train_data (DataFrame): The training data.

    Returns:
    (RandomForestClassificationModel, dict): The best model and its hyperparameters.
    """
    rf = RandomForestClassifier(featuresCol='scaledFeatures', labelCol='label')
    paramGrid = (ParamGridBuilder()
                 .addGrid(rf.numTrees, [10, 20, 30])
                 .addGrid(rf.maxDepth, [5, 10, 15])
                 .build())

    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")

    cv = CrossValidator(estimator=rf, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=3)
    cv_model = cv.fit(train_data)

    best_model = cv_model.bestModel
    best_params = {param.name: best_model.getOrDefault(param) for param in paramGrid[0]}
    
    return best_model, best_params
