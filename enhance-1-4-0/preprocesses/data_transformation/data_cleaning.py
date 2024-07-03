from pyspark.sql import DataFrame
from enhance.Enhance.utils.logger import get_logger
from enhance.Enhance.utils.exception_handler import handle_exceptions

logger = get_logger(__name__)

class DataCleaner:
    """
    Class for performing basic data cleaning operations such as removing nulls, filling nulls, and removing duplicates.
    """

    @staticmethod
    @handle_exceptions
    def remove_nulls(df: DataFrame, columns: list) -> DataFrame:
        """
        Remove rows with null values in specified columns.

        Args:
            df (DataFrame): DataFrame containing the data.
            columns (list): List of columns to check for null values.

        Returns:
            DataFrame: DataFrame with rows containing null values removed.
        """
        logger.info(f"Removing nulls from columns: {columns}")
        return df.dropna(subset=columns)

    @staticmethod
    @handle_exceptions
    def fill_nulls(df: DataFrame, columns: dict) -> DataFrame:
        """
        Fill null values in specified columns with given values.

        Args:
            df (DataFrame): DataFrame containing the data.
            columns (dict): Dictionary with column names as keys and fill values as values.

        Returns:
            DataFrame: DataFrame with null values filled.
        """
        logger.info(f"Filling nulls in columns: {columns}")
        for column, value in columns.items():
            try:
                df = df.fillna({column: value})
            except Exception as e:
                logger.error(f"Error filling nulls in column {column} with value {value}: {str(e)}", exc_info=True)
                raise e
        return df

    @staticmethod
    @handle_exceptions
    def remove_duplicates(df: DataFrame) -> DataFrame:
        """
        Remove duplicate rows in DataFrame.

        Args:
            df (DataFrame): DataFrame containing the data.

        Returns:
            DataFrame: DataFrame with duplicate rows removed.
        """
        logger.info("Removing duplicate rows")
        try:
            df = df.dropDuplicates()
        except Exception as e:
            logger.error(f"Error removing duplicates: {str(e)}", exc_info=True)
            raise e
        return df