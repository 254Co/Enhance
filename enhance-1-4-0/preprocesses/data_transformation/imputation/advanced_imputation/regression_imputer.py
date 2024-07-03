# regression_imputer.py

from pyspark.sql import DataFrame
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline

class SparkRegressionImputer:
    @staticmethod
    def impute(data: DataFrame, target_col: str, feature_cols: list) -> DataFrame:
        data = data.na.drop(subset=feature_cols)
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
        lr = LinearRegression(featuresCol="features", labelCol=target_col)
        pipeline = Pipeline(stages=[assembler, lr])
        model = pipeline.fit(data)

        missing_data = data.filter(data[target_col].isNull())
        predictions = model.transform(missing_data)
        data = data.na.drop(subset=[target_col])
        data = data.union(predictions.select(data.columns))
        return data
