# Enhance/processes/ML/DL/RNNs_LSTMs/common/training_utils.py

from pyspark.sql import DataFrame
from enhance.Enhance.utils.logger import get_logger
from enhance.Enhance.utils.exception_handler import handle_exceptions
from enhance.Enhance.data_processing.data_transformation import (
    DataCleaner, FeatureEngineer, DataTransformer, DataImputer,
    OutlierRemover, CategoricalEncoder, DatetimeFeatureExtractor, ParallelProcessingUtils
)

logger = get_logger(__name__)

class TrainingUtils:

    @staticmethod
    @handle_exceptions
    def preprocess_data(df: DataFrame, cleaning_params: dict, feature_params: dict, transformation_params: dict) -> DataFrame:
        """
        Preprocess the data using advanced data transformation tools.
        """
        logger.info("Data preprocessing started")
        cleaner = DataCleaner()
        engineer = FeatureEngineer()
        transformer = DataTransformer()
        imputer = DataImputer()
        outlier_remover = OutlierRemover()
        encoder = CategoricalEncoder()
        datetime_extractor = DatetimeFeatureExtractor()

        # Data Cleaning
        df = cleaner.remove_nulls(df, cleaning_params.get('remove_nulls', []))
        df = cleaner.fill_nulls(df, cleaning_params.get('fill_nulls', {}))
        df = cleaner.remove_duplicates(df)

        # Data Imputation
        df = imputer.impute_with_mode(df, cleaning_params.get('impute_with_mode', []))
        df = imputer.impute_with_constant(df, cleaning_params.get('impute_with_constant', {}))

        # Outlier Removal
        df = outlier_remover.z_score_method(df, cleaning_params.get('z_score_columns', []), cleaning_params.get('z_score_threshold', 3.0))

        # Categorical Encoding
        df = encoder.one_hot_encode(df, feature_params.get('one_hot_encode', []))
        df = encoder.label_encode(df, feature_params.get('label_encode', []))

        # Datetime Feature Extraction
        df = datetime_extractor.extract_features(df, feature_params.get('datetime_column', ''))

        # Feature Engineering
        for new_col, params in feature_params.items():
            if new_col not in ['one_hot_encode', 'label_encode', 'datetime_column']:
                df = engineer.add_feature(df, new_col, params['calculation_udf'], params['feature_columns'])

        # Data Transformation
        df = transformer.normalize(df, transformation_params.get('input_cols', []), transformation_params.get('output_col', 'normalized_features'), transformation_params.get('method', 'min-max'))

        logger.info("Data preprocessing completed")
        return df

    @staticmethod
    @handle_exceptions
    def parallel_preprocess_data(df: DataFrame, cleaning_params: dict, feature_params: dict, transformation_params: dict) -> DataFrame:
        """
        Parallelize the data preprocessing steps.
        """
        logger.info("Parallel preprocessing started")
        parallel_utils = ParallelProcessingUtils()

        def processing_func(batch_df: DataFrame) -> DataFrame:
            # Reusing the existing preprocessing logic for each partition
            return TrainingUtils.preprocess_data(batch_df, cleaning_params, feature_params, transformation_params)

        result_df = parallel_utils.parallelize_dataframe_processing(df, processing_func)
        logger.info("Parallel preprocessing completed")
        return result_df