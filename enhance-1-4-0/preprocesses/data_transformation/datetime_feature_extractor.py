from pyspark.sql import DataFrame
from pyspark.sql.functions import (year, month, dayofmonth, hour, minute, second, quarter, dayofweek, weekofyear)
from enhance.Enhance.utils.logger import get_logger
from enhance.Enhance.utils.exception_handler import handle_exceptions

logger = get_logger(__name__)

class DatetimeFeatureExtractor:
    """
    Class for extracting datetime features from a specified datetime column into separate columns.
    """

    @staticmethod
    @handle_exceptions
    def extract_features(df: DataFrame, column: str) -> DataFrame:
        """
        Extracts datetime features from the specified datetime column into separate columns.

        Args:
            df (DataFrame): DataFrame containing the data.
            column (str): The name of the datetime column to extract features from.

        Returns:
            DataFrame: DataFrame with extracted datetime features.
        """
        logger.info(f"Extracting datetime features from column: {column}")
        try:
            df = df.withColumn(column + "_year", year(df[column]))
            df = df.withColumn(column + "_month", month(df[column]))
            df = df.withColumn(column + "_day", dayofmonth(df[column]))
            df = df.withColumn(column + "_hour", hour(df[column]))
            df = df.withColumn(column + "_minute", minute(df[column]))
            df = df.withColumn(column + "_second", second(df[column]))
            df = df.withColumn(column + "_quarter", quarter(df[column]))
            df = df.withColumn(column + "_dayofweek", dayofweek(df[column]))
            df = df.withColumn(column + "_weekofyear", weekofyear(df[column]))
        except Exception as e:
            logger.error(f"Error extracting datetime features from column {column}: {str(e)}", exc_info=True)
            raise e
        return df