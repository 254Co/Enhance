import numpy as np
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, udf
from pyspark.sql.types import DoubleType
from enhance.Enhance.utils.logger import get_logger
from enhance.Enhance.utils.exception_handler import handle_exceptions

logger = get_logger(__name__)

def create_time_series_features(data, look_back=1):
    X, y = [], []
    for i in range(len(data) - look_back - 1):
        a = data[i:(i + look_back), 0]
        X.append(a)
        y.append(data[i + look_back, 0])
    return np.array(X), np.array(y)

def generate_lstm_inputs(data, look_back):
    X, y = create_time_series_features(data, look_back)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y

class FeatureEngineer:

    @staticmethod
    @handle_exceptions
    def add_feature(df: DataFrame, new_column: str, calculation_udf, feature_columns: list) -> DataFrame:
        """
        Add a new feature column based on UDF calculation.
        """
        logger.info(f"Adding new feature column: {new_column} based on columns: {feature_columns}")
        calc_udf = udf(calculation_udf, DoubleType())
        return df.withColumn(new_column, calc_udf(*[col(c) for c in feature_columns]))

    @staticmethod
    @handle_exceptions
    def scale_features(df: DataFrame, columns: list, scaler) -> DataFrame:
        """
        Scale specified feature columns using provided scaler.
        """
        logger.info(f"Scaling features: {columns}")
        for column in columns:
            df = df.withColumn(column, scaler(df[column]))
        return df

    @staticmethod
    @handle_exceptions
    def polynomial_features(df: DataFrame, columns: list, degree: int = 2) -> DataFrame:
        """
        Generate polynomial features for specified columns.
        """
        logger.info(f"Generating polynomial features for columns: {columns} up to degree: {degree}")
        from pyspark.ml.feature import PolynomialExpansion, VectorAssembler

        assembler = VectorAssembler(inputCols=columns, outputCol="features")
        df = assembler.transform(df)
        polyExpansion = PolynomialExpansion(degree=degree, inputCol="features", outputCol="poly_features")
        df = polyExpansion.transform(df).drop("features")
        return df