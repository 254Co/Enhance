"""
Support Vector Machine (SVM) training and tuning module using Spark MLlib.

Author: 254StudioZ LLC
Date: 2024-07-01
"""

from pyspark.ml.classification import LinearSVC, LinearSVCModel
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import DataFrame

def train_svm(train_data: DataFrame) -> LinearSVCModel:
    """
    Trains a SVM model on the given training data.

    Parameters:
    train_data (DataFrame): The training data.

    Returns:
    LinearSVCModel: The trained SVM model.
    """
    svm = LinearSVC(featuresCol='scaledFeatures', labelCol='label')
    model = svm.fit(train_data)
    return model

def tune_svm(train_data: DataFrame) -> (LinearSVCModel, dict):
    """
    Tunes a SVM model using CrossValidator to find the best hyperparameters.

    Parameters:
    train_data (DataFrame): The training data.

    Returns:
    (LinearSVCModel, dict): The best model and its hyperparameters.
    """
    svm = LinearSVC(featuresCol='scaledFeatures', labelCol='label')
    paramGrid = (ParamGridBuilder()
                 .addGrid(svm.regParam, [0.01, 0.1, 1.0])
                 .addGrid(svm.maxIter, [10, 50, 100])
                 .build())

    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")

    cv = CrossValidator(estimator=svm, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=3)
    cv_model = cv.fit(train_data)

    best_model = cv_model.bestModel
    best_params = {param.name: best_model.getOrDefault(param) for param in paramGrid[0]}
    
    return best_model, best_params
