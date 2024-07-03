# knn_imputer.py

from pyspark.sql import DataFrame
from pyspark.ml.feature import Imputer

class SparkKNNImputer:
    @staticmethod
    def impute(data: DataFrame, columns: list = None, k: int = 5) -> DataFrame:
        if columns is None:
            columns = data.columns
        imputer = Imputer(inputCols=columns, outputCols=columns)
        model = imputer.fit(data)
        data = model.transform(data)
        return data
