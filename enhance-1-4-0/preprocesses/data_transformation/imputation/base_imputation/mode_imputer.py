# mode_imputer.py

from pyspark.sql import DataFrame
from pyspark.sql.functions import col, count

class SparkModeImputer:
    @staticmethod
    def impute(data: DataFrame, columns: list = None) -> DataFrame:
        if columns is None:
            columns = data.columns
        for col_name in columns:
            mode_value = data.groupBy(col_name).count().orderBy('count', ascending=False).first()[0]
            data = data.na.fill({col_name: mode_value})
        return data
