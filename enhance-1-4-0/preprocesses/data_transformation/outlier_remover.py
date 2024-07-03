from pyspark.sql import DataFrame
from pyspark.sql.functions import col
from enhance.Enhance.utils.logger import get_logger
from enhance.Enhance.utils.exception_handler import handle_exceptions

logger = get_logger(__name__)

class OutlierRemover:
    """
    Class for removing outliers from specified columns using different methods such as z-score and IQR.
    """

    @staticmethod
    @handle_exceptions
    def z_score_method(df: DataFrame, columns: list, threshold: float = 3.0) -> DataFrame:
        """
        Remove outliers based on z-score method.

        Args:
            df (DataFrame): DataFrame containing the data.
            columns (list): List of columns to remove outliers from.
            threshold (float): Z-score threshold for identifying outliers.

        Returns:
            DataFrame: DataFrame with outliers removed.
        """
        logger.info(f"Removing outliers in columns: {columns} using z-score method with threshold: {threshold}")
        from pyspark.sql import functions as F
        for column in columns:
            try:
                mean = df.agg({column: 'mean'}).collect()[0][0]
                stddev = df.agg({column: 'stddev'}).collect()[0][0]
                df = df.filter((F.col(column) - mean) / stddev <= threshold)
            except Exception as e:
                logger.error(f"Error removing outliers from column {column} using z-score method: {str(e)}", exc_info=True)
                raise e
        return df

    @staticmethod
    @handle_exceptions
    def iqr_method(df: DataFrame, columns: list) -> DataFrame:
        """
        Remove outliers based on IQR method.

        Args:
            df (DataFrame): DataFrame containing the data.
            columns (list): List of columns to remove outliers from.

        Returns:
            DataFrame: DataFrame with outliers removed.
        """
        logger.info(f"Removing outliers in columns: {columns} using IQR method")
        from pyspark.sql import functions as F
        for column in columns:
            try:
                quantiles = df.approxQuantile(column, [0.25, 0.75], 0.05)
                Q1, Q3 = quantiles[0], quantiles[1]
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df = df.filter((col(column) >= lower_bound) & (col(column) <= upper_bound))
            except Exception as e:
                logger.error(f"Error removing outliers from column {column} using IQR method: {str(e)}", exc_info=True)
                raise e
        return df