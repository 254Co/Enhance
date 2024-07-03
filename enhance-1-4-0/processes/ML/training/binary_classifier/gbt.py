"""
Gradient-Boosted Trees training and tuning module using Spark MLlib.

Author: 254StudioZ LLC
Date: 2024-07-01
"""

from pyspark.ml.classification import GBTClassifier, GBTClassificationModel
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import DataFrame

def train_gbt(train_data: DataFrame) -> GBTClassificationModel:
    """
    Trains a Gradient-Boosted Trees model on the given training data.

    Parameters:
    train_data (DataFrame): The training data.

    Returns:
    GBTClassificationModel: The trained Gradient-Boosted Trees model.
    """
    gbt = GBTClassifier(featuresCol='scaledFeatures', labelCol='label')
    model = gbt.fit(train_data)
    return model

def tune_gbt(train_data: DataFrame) -> (GBTClassificationModel, dict):
    """
    Tunes a Gradient-Boosted Trees model using CrossValidator to find the best hyperparameters.

    Parameters:
    train_data (DataFrame): The training data.

    Returns:
    (GBTClassificationModel, dict): The best model and its hyperparameters.
    """
    gbt = GBTClassifier(featuresCol='scaledFeatures', labelCol='label')
    paramGrid = (ParamGridBuilder()
                 .addGrid(gbt.maxDepth, [5, 10, 15])
                 .addGrid(gbt.maxIter, [10, 20, 30])
                 .build())

    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")

    cv = CrossValidator(estimator=gbt, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=3)
    cv_model = cv.fit(train_data)

    best_model = cv_model.bestModel
    best_params = {param.name: best_model.getOrDefault(param) for param in paramGrid[0]}
    
    return best_model, best_params
