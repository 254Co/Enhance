from pyspark.sql import DataFrame
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from enhance.Enhance.utils.logger import get_logger
from enhance.Enhance.utils.exception_handler import handle_exceptions

logger = get_logger(__name__)

class CategoricalEncoder:
    """
    Class for performing categorical encoding such as One Hot Encoding and Label Encoding on specified columns.
    """

    @staticmethod
    @handle_exceptions
    def one_hot_encode(df: DataFrame, columns: list) -> DataFrame:
        """
        One Hot Encode specified columns.

        Args:
            df (DataFrame): DataFrame containing the data.
            columns (list): List of columns to be one hot encoded.

        Returns:
            DataFrame: DataFrame with one hot encoded columns.
        """
        logger.info(f"One hot encoding columns: {columns}")
        for column in columns:
            try:
                indexer = StringIndexer(inputCol=column, outputCol=column + "_index")
                encoder = OneHotEncoder(inputCol=column + "_index", outputCol=column + "_onehot")
                df = indexer.fit(df).transform(df)
                df = encoder.fit(df).transform(df)
            except Exception as e:
                logger.error(f"Error one hot encoding column {column}: {str(e)}", exc_info=True)
                raise e
        return df

    @staticmethod
    @handle_exceptions
    def label_encode(df: DataFrame, columns: list) -> DataFrame:
        """
        Label Encode specified columns.

        Args:
            df (DataFrame): DataFrame containing the data.
            columns (list): List of columns to be label encoded.

        Returns:
            DataFrame: DataFrame with label encoded columns.
        """
        logger.info(f"Label encoding columns: {columns}")
        for column in columns:
            try:
                indexer = StringIndexer(inputCol=column, outputCol=column + "_label")
                df = indexer.fit(df).transform(df)
            except Exception as e:
                logger.error(f"Error label encoding column {column}: {str(e)}", exc_info=True)
                raise e
        return df