from pyspark.sql import DataFrame
from pyspark.sql.functions import col, expr
from enhance.Enhance.utils.logger import get_logger
from enhance.Enhance.utils.exception_handler import handle_exceptions

logger = get_logger(__name__)

class AdvancedDataCleaner:
    """
    Class for performing advanced data cleaning operations such as imputing missing values and removing outliers.
    """

    @staticmethod
    @handle_exceptions
    def impute_missing_values(df: DataFrame, columns: dict, strategy: str = 'mean') -> DataFrame:
        """
        Impute missing values in specified columns using given strategy.

        Args:
            df (DataFrame): DataFrame containing the data.
            columns (dict): Dictionary with column names as keys and impute values as values.
            strategy (str): Strategy for imputing missing values, either 'mean' or 'median'.

        Returns:
            DataFrame: DataFrame with imputed values.
        """
        logger.info(f"Imputing missing values in columns: {columns} using strategy: {strategy}")
        for column in columns:
            try:
                if strategy == 'mean':
                    impute_value = df.selectExpr(f"mean({column}) as mean").collect()[0]['mean']
                elif strategy == 'median':
                    impute_value = df.approxQuantile(column, [0.5], 0.25)[0]
                else:
                    raise ValueError("Unsupported imputation strategy")
                df = df.fillna({column: impute_value})
            except Exception as e:
                logger.error(f"Error imputing column {column}: {str(e)}", exc_info=True)
                raise e
        return df

    @staticmethod
    @handle_exceptions
    def remove_outliers(df: DataFrame, columns: list, lower_bound: float = 0.05, upper_bound: float = 0.95) -> DataFrame:
        """
        Remove outliers in specified columns based on quantile bounds.

        Args:
            df (DataFrame): DataFrame containing the data.
            columns (list): List of columns from which to remove outliers.
            lower_bound (float): Lower quantile bound.
            upper_bound (float): Upper quantile bound.

        Returns:
            DataFrame: DataFrame with outliers removed.
        """
        logger.info(f"Removing outliers in columns: {columns} using bounds: {lower_bound}, {upper_bound}")
        for column in columns:
            try:
                bounds = df.selectExpr(f"percentile_approx({column}, array({lower_bound}, {upper_bound})) as bounds").collect()[0]['bounds']
                lower, upper = bounds[0], bounds[1]
                df = df.filter((col(column) >= lower) & (col(column) <= upper))
            except Exception as e:
                logger.error(f"Error removing outliers from column {column}: {str(e)}", exc_info=True)
                raise e
        return df