# mice_imputer.py

from pyspark.sql import DataFrame
from pyspark.ml.feature import Imputer

class SparkMICEImputer:
    @staticmethod
    def impute(data: DataFrame, columns: list = None, max_iter: int = 10) -> DataFrame:
        if columns is None:
            columns = data.columns
        imputer = Imputer(inputCols=columns, outputCols=columns, strategy='mean')
        model = imputer.fit(data)
        data = model.transform(data)
        return data
