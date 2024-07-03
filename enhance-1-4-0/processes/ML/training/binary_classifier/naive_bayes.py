"""
Naive Bayes training and tuning module using Spark MLlib.

Author: 254StudioZ LLC
Date: 2024-07-01
"""

from pyspark.ml.classification import NaiveBayes, NaiveBayesModel
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import DataFrame

def train_naive_bayes(train_data: DataFrame) -> NaiveBayesModel:
    """
    Trains a Naive Bayes model on the given training data.

    Parameters:
    train_data (DataFrame): The training data.

    Returns:
    NaiveBayesModel: The trained Naive Bayes model.
    """
    nb = NaiveBayes(featuresCol='scaledFeatures', labelCol='label')
    model = nb.fit(train_data)
    return model

def tune_naive_bayes(train_data: DataFrame) -> (NaiveBayesModel, dict):
    """
    Tunes a Naive Bayes model using CrossValidator to find the best hyperparameters.

    Parameters:
    train_data (DataFrame): The training data.

    Returns:
    (NaiveBayesModel, dict): The best model and its hyperparameters.
    """
    nb = NaiveBayes(featuresCol='scaledFeatures', labelCol='label')
    paramGrid = (ParamGridBuilder()
                 .addGrid(nb.smoothing, [0.5, 1.0, 1.5])
                 .build())

    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")

    cv = CrossValidator(estimator=nb, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=3)
    cv_model = cv.fit(train_data)

    best_model = cv_model.bestModel
    best_params = {param.name: best_model.getOrDefault(param) for param in paramGrid[0]}
    
    return best_model, best_params
