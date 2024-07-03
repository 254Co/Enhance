# interpolation_imputer.py

from pyspark.sql import DataFrame
from pyspark.sql.functions import pandas_udf, PandasUDFType
import pandas as pd

class SparkInterpolationImputer:
    @staticmethod
    def impute(data: DataFrame, columns: list = None, method: str = 'linear') -> DataFrame:
        if columns is None:
            columns = data.columns

        @pandas_udf("double", PandasUDFType.SCALAR)
        def interpolate(column: pd.Series) -> pd.Series:
            return column.interpolate(method=method)

        for col in columns:
            data = data.withColumn(col, interpolate(data[col]))
        return data
