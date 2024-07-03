# bfill_imputer.py

from pyspark.sql import DataFrame
from pyspark.sql.window import Window
from pyspark.sql.functions import first

class SparkBFillImputer:
    @staticmethod
    def impute(data: DataFrame, columns: list = None) -> DataFrame:
        if columns is None:
            columns = data.columns
        for col in columns:
            window_spec = Window.orderBy('index').rowsBetween(0, sys.maxsize)
            data = data.withColumn(col, first(col, True).over(window_spec))
        return data
