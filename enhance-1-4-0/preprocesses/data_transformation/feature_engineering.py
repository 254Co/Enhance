from pyspark.sql import DataFrame
from pyspark.sql.functions import col, udf
from pyspark.sql.types import DoubleType
from enhance.Enhance.utils.logger import get_logger
from enhance.Enhance.utils.exception_handler import handle_exceptions

logger = get_logger(__name__)

class FeatureEngineer:
    """
    Class for performing feature engineering operations such as adding new features, scaling features, and generating polynomial features.
    """

    @staticmethod
    @handle_exceptions
    def add_feature(df: DataFrame, new_column: str, calculation_udf, feature_columns: list) -> DataFrame:
        """
        Add a new feature column based on UDF calculation.

        Args:
            df (DataFrame): DataFrame containing the data.
            new_column (str): Name of the new feature column.
            calculation_udf: User Defined Function for calculating the new feature.
            feature_columns (list): List of columns to be used in the calculation.

        Returns:
            DataFrame: DataFrame with the new feature column added.
        """
        logger.info(f"Adding new feature column: {new_column} based on columns: {feature_columns}")
        calc_udf = udf(calculation_udf, DoubleType())
        return df.withColumn(new_column, calc_udf(*[col(c) for c in feature_columns]))

    @staticmethod
    @handle_exceptions
    def scale_features(df: DataFrame, columns: list, scaler) -> DataFrame:
        """
        Scale specified feature columns using provided scaler.

        Args:
            df (DataFrame): DataFrame containing the data.
            columns (list): List of columns to be scaled.
            scaler: Scaler function to be applied to the columns.

        Returns:
            DataFrame: DataFrame with scaled features.
        """
        logger.info(f"Scaling features: {columns}")
        for column in columns:
            try:
                df = df.withColumn(column, scaler(df[column]))
            except Exception as e:
                logger.error(f"Error scaling column {column}: {str(e)}", exc_info=True)
                raise e
        return df

    @staticmethod
    @handle_exceptions
    def polynomial_features(df: DataFrame, columns: list, degree: int = 2) -> DataFrame:
        """
        Generate polynomial features for specified columns.

        Args:
            df (DataFrame): DataFrame containing the data.
            columns (list): List of columns to generate polynomial features for.
            degree (int): Degree of the polynomial features.

        Returns:
            DataFrame: DataFrame with polynomial features added.
        """
        logger.info(f"Generating polynomial features for columns: {columns} up to degree: {degree}")
        from pyspark.ml.feature import PolynomialExpansion, VectorAssembler

        try:
            assembler = VectorAssembler(inputCols=columns, outputCol="features")
            df = assembler.transform(df)
            polyExpansion = PolynomialExpansion(degree=degree, inputCol="features", outputCol="poly_features")
            df = polyExpansion.transform(df).drop("features")
        except Exception as e:
            logger.error(f"Error generating polynomial features: {str(e)}", exc_info=True)
            raise e
        return df