# mean_imputer.py

from pyspark.sql import DataFrame
from pyspark.sql.functions import mean

class SparkMeanImputer:
    @staticmethod
    def impute(data: DataFrame, columns: list = None) -> DataFrame:
        if columns is None:
            columns = data.columns
        for col in columns:
            mean_value = data.select(mean(col)).collect()[0][0]
            data = data.na.fill({col: mean_value})
        return data
