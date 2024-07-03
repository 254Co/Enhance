from enhance.Enhance.utils.logger import get_logger
from enhance.Enhance.utils.exception_handler import handle_exceptions
from pyspark.sql import DataFrame
from pyspark.ml.feature import VectorAssembler, VarianceThresholdSelector, ChiSqSelector
from pyspark.ml.stat import Correlation
from pyspark.ml.linalg import Vectors
import logging

logger = get_logger(__name__)

class FeatureSelector:
    """
    Feature Selector for selecting important features.
    """
    def __init__(self, method='variance', threshold=0.0, num_top_features=50):
        self.method = method
        self.threshold = threshold
        self.num_top_features = num_top_features
        logger.info(f"FeatureSelector initialized with method: {method}, threshold: {threshold}, num_top_features: {num_top_features}")

    @handle_exceptions
    def fit(self, df: DataFrame, label_col: str, feature_cols: list):
        """
        Fit the selector to the data.
        """
        logger.info(f"Fitting FeatureSelector to data with method: {self.method}")
        if self.method == 'variance':
            assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
            df = assembler.transform(df)
            selector = VarianceThresholdSelector(varianceThreshold=self.threshold, outputCol="selectedFeatures")
            self.model = selector.fit(df)
        elif self.method == 'chi2':
            assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
            df = assembler.transform(df)
            selector = ChiSqSelector(numTopFeatures=self.num_top_features, featuresCol="features", labelCol=label_col, outputCol="selectedFeatures")
            self.model = selector.fit(df)
        elif self.method == 'correlation':
            assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
            df = assembler.transform(df)
            matrix = Correlation.corr(df, "features").head()[0]
            self.model = matrix
        else:
            raise ValueError(f"Unsupported feature selection method: {self.method}")
        logger.info("FeatureSelector fitting completed")

    @handle_exceptions
    def transform(self, df: DataFrame):
        """
        Transform the data using the fitted selector.
        """
        logger.info(f"Transforming data using FeatureSelector with method: {self.method}")
        if self.method in ['variance', 'chi2']:
            return self.model.transform(df)
        elif self.method == 'correlation':
            # Implement correlation-based feature selection transformation
            # This is a placeholder for actual implementation
            logger.info("Correlation-based feature selection transformation is not implemented yet")
            return df
        else:
            raise ValueError(f"Unsupported feature selection method: {self.method}")

    @handle_exceptions
    def fit_transform(self, df: DataFrame, label_col: str, feature_cols: list):
        """
        Fit the selector to the data and transform it.
        """
        logger.info("Calling fit_transform on FeatureSelector")
        self.fit(df, label_col, feature_cols)
        return self.transform(df)