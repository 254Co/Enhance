# median_imputer.py

from pyspark.sql import DataFrame

class SparkMedianImputer:
    @staticmethod
    def impute(data: DataFrame, columns: list = None) -> DataFrame:
        if columns is None:
            columns = data.columns
        for col in columns:
            median_value = data.approxQuantile(col, [0.5], 0.01)[0]
            data = data.na.fill({col: median_value})
        return data
