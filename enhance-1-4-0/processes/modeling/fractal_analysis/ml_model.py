# ml_model.py
from pyspark.sql import DataFrame
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

def build_ml_model(data: DataFrame) -> dict:
    """
    Build a machine learning model to predict future prices based on fractal dimensions and other features.
    
    Parameters:
    - data: Input Spark DataFrame containing market data and calculated features.
    
    Returns:
    - dict: Dictionary containing the trained model and evaluation metrics.
    """
    
    # Assemble features into a feature vector
    feature_columns = [col for col in data.columns if col.startswith("rolling_") or col.startswith("fractal_")]
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    data = assembler.transform(data)
    
    # Split the data into training and test sets
    train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)
    
    # Build and train the linear regression model
    lr = LinearRegression(featuresCol="features", labelCol="price", predictionCol="prediction")
    lr_model = lr.fit(train_data)
    
    # Evaluate the model
    evaluator = RegressionEvaluator(labelCol="price", predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(lr_model.transform(test_data))
    
    return {"model": lr_model, "rmse": rmse}
