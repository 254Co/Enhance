"""
Decision Tree training and tuning module using Spark MLlib.

Author: 254StudioZ LLc
Date: 2024-07-01
"""

from pyspark.ml.classification import DecisionTreeClassifier, DecisionTreeClassificationModel
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import DataFrame

def train_decision_tree(train_data: DataFrame) -> DecisionTreeClassificationModel:
    """
    Trains a Decision Tree model on the given training data.

    Parameters:
    train_data (DataFrame): The training data.

    Returns:
    DecisionTreeClassificationModel: The trained Decision Tree model.
    """
    dt = DecisionTreeClassifier(featuresCol='scaledFeatures', labelCol='label')
    model = dt.fit(train_data)
    return model

def tune_decision_tree(train_data: DataFrame) -> (DecisionTreeClassificationModel, dict):
    """
    Tunes a Decision Tree model using CrossValidator to find the best hyperparameters.

    Parameters:
    train_data (DataFrame): The training data.

    Returns:
    (DecisionTreeClassificationModel, dict): The best model and its hyperparameters.
    """
    dt = DecisionTreeClassifier(featuresCol='scaledFeatures', labelCol='label')
    paramGrid = (ParamGridBuilder()
                 .addGrid(dt.maxDepth, [5, 10, 15])
                 .addGrid(dt.impurity, ['gini', 'entropy'])
                 .build())

    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")

    cv = CrossValidator(estimator=dt, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=3)
    cv_model = cv.fit(train_data)

    best_model = cv_model.bestModel
    best_params = {param.name: best_model.getOrDefault(param) for param in paramGrid[0]}
    
    return best_model, best_params
