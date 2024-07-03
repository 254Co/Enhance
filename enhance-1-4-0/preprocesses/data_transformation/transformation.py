from pyspark.sql import DataFrame
from pyspark.ml.feature import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, VectorAssembler
from enhance.Enhance.utils.logger import get_logger
from enhance.Enhance.utils.exception_handler import handle_exceptions

logger = get_logger(__name__)

class DataTransformer:
    """
    Class for performing data transformation operations such as normalization using various scaling methods.
    """

    @staticmethod
    @handle_exceptions
    def normalize(df: DataFrame, input_cols: list, output_col: str, method: str = 'min-max') -> DataFrame:
        """
        Normalize specified columns.

        Args:
            df (DataFrame): DataFrame containing the data.
            input_cols (list): List of columns to be normalized.
            output_col (str): Name of the output column for normalized features.
            method (str): Normalization method, one of 'min-max', 'standard', 'max-abs', 'robust'.

        Returns:
            DataFrame: DataFrame with normalized columns.
        """
        logger.info(f"Normalizing columns: {input_cols} using method: {method}")
        assembler = VectorAssembler(inputCols=input_cols, outputCol='features_assembled')
        df = assembler.transform(df)

        if method == 'min-max':
            scaler = MinMaxScaler(inputCol="features_assembled", outputCol=output_col)
        elif method == 'standard':
            scaler = StandardScaler(inputCol="features_assembled", outputCol=output_col)
        elif method == 'max-abs':
            scaler = MaxAbsScaler(inputCol="features_assembled", outputCol=output_col)
        elif method == 'robust':
            scaler = RobustScaler(inputCol="features_assembled", outputCol=output_col)
        else:
            raise ValueError(f"Unsupported normalization method: {method}")

        scaler_model = scaler.fit(df)
        return scaler_model.transform(df).drop('features_assembled')