# ffill_imputer.py

from pyspark.sql import DataFrame
from pyspark.sql.window import Window
from pyspark.sql.functions import last

class SparkFFillImputer:
    @staticmethod
    def impute(data: DataFrame, columns: list = None) -> DataFrame:
        if columns is None:
            columns = data.columns
        for col in columns:
            window_spec = Window.orderBy('index').rowsBetween(-sys.maxsize, 0)
            data = data.withColumn(col, last(col, True).over(window_spec))
        return data
