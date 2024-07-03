from pyspark.sql.functions import pandas_udf, PandasUDFType
from enhance.Enhance.utils.logger import get_logger
from enhance.Enhance.utils.exception_handler import handle_exceptions
from enhance.Enhance.data_processing.data_transformation.utils import ParallelProcessingUtils
from enhance.Enhance.data_processing.data_transformation import (
    DataCleaner, FeatureEngineer, DataTransformer, DataImputer,
    OutlierRemover, CategoricalEncoder, DatetimeFeatureExtractor
)

logger = get_logger(__name__)

class Trainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        logger.info(f"Trainer initialized with model: {model} and config: {config}")

    @handle_exceptions
    def train_model(self, sequences, labels):
        import numpy as np
        sequences = np.array(sequences.tolist())
        labels = np.array(labels.tolist())
        self.model.fit(sequences, labels, epochs=self.config.EPOCHS)
        self.model.save(self.config.MODEL_PATH)
        logger.info("Model training completed and saved")

    def train_model_udf(self):
        return pandas_udf(self.train_model, "binary", PandasUDFType.SCALAR)

    @handle_exceptions
    def train(self, train_df):
        train_model_udf = self.train_model_udf()
        train_model_udf(train_df.select("sequences"), train_df.select("label"))
        logger.info("Training process completed")

    @handle_exceptions
    def preprocess_data(self, df, cleaning_params, feature_params, transformation_params):
        logger.info("Data preprocessing started")
        cleaner = DataCleaner()
        engineer = FeatureEngineer()
        transformer = DataTransformer()
        imputer = DataImputer()
        outlier_remover = OutlierRemover()
        encoder = CategoricalEncoder()
        datetime_extractor = DatetimeFeatureExtractor()

        df = cleaner.remove_nulls(df, cleaning_params.get('remove_nulls', []))
        df = cleaner.fill_nulls(df, cleaning_params.get('fill_nulls', {}))
        df = cleaner.remove_duplicates(df)

        df = imputer.impute_with_mode(df, cleaning_params.get('impute_with_mode', []))
        df = imputer.impute_with_constant(df, cleaning_params.get('impute_with_constant', {}))

        df = outlier_remover.z_score_method(df, cleaning_params.get('z_score_columns', []), cleaning_params.get('z_score_threshold', 3.0))

        df = encoder.one_hot_encode(df, feature_params.get('one_hot_encode', []))
        df = encoder.label_encode(df, feature_params.get('label_encode', []))

        df = datetime_extractor.extract_features(df, feature_params.get('datetime_column', ''))

        for new_col, params in feature_params.items():
            if new_col not in ['one_hot_encode', 'label_encode', 'datetime_column']:
                df = engineer.add_feature(df, new_col, params['calculation_udf'], params['feature_columns'])

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
            trainer = Trainer(model=None, config=None)  # Dummy Trainer for using the method
            return trainer.preprocess_data(batch_df, cleaning_params, feature_params, transformation_params)

        result_df = parallel_utils.parallelize_dataframe_processing(df, processing_func)
        logger.info("Parallel preprocessing completed")
        return result_df