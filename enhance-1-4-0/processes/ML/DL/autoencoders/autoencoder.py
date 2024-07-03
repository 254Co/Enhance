# autoencoder.py
import tensorflow as tf
from tensorflow.keras import layers, models
from Enhance.data_processing.data_transformation import DataCleaner, FeatureEngineer, DataTransformer, DataImputer, OutlierRemover, CategoricalEncoder, DatetimeFeatureExtractor
from Enhance.utils.logger import get_logger
from Enhance.utils.exception_handler import handle_exceptions

logger = get_logger(__name__)

class Autoencoder:
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.model = self.build_model()
        self.encoder = None

    def build_model(self):
        input_layer = layers.Input(shape=(self.input_dim,))
        encoded = layers.Dense(128, activation='relu')(input_layer)
        encoded = layers.Dense(64, activation='relu')(encoded)
        encoded = layers.Dense(32, activation='relu')(encoded)

        decoded = layers.Dense(64, activation='relu')(encoded)
        decoded = layers.Dense(128, activation='relu')(decoded)
        decoded = layers.Dense(self.input_dim, activation='sigmoid')(decoded)

        autoencoder = models.Model(inputs=input_layer, outputs=decoded)
        autoencoder.compile(optimizer='adam', loss='mse')

        self.encoder = models.Model(inputs=input_layer, outputs=encoded)

        return autoencoder

    @handle_exceptions
    def train_batch(self, x_train, batch_size=256):
        logger.info("Training batch started")
        self.model.fit(x_train, x_train, batch_size=batch_size, epochs=1, verbose=0)
        logger.info("Training batch completed")

    @handle_exceptions
    def dimensionality_reduction(self, x):
        logger.info("Dimensionality reduction started")
        reduced_dimensions = self.encoder.predict(x)
        logger.info("Dimensionality reduction completed")
        return reduced_dimensions

    @handle_exceptions
    def anomaly_detection(self, x, threshold):
        logger.info("Anomaly detection started")
        reconstructions = self.model.predict(x)
        loss = tf.keras.losses.mse(reconstructions, x)
        anomalies = loss > threshold
        logger.info("Anomaly detection completed")
        return anomalies

    @handle_exceptions
    def noise_reduction(self, x_noisy):
        logger.info("Noise reduction started")
        denoised_data = self.model.predict(x_noisy)
        logger.info("Noise reduction completed")
        return denoised_data

    @handle_exceptions
    def save_model(self, bucket_name, model_name):
        import tempfile
        from .gcs_utils import save_model_to_gcs
        logger.info(f"Saving model to bucket: {bucket_name}, model name: {model_name}")
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = f"{temp_dir}/{model_name}"
            self.model.save(model_path)
            save_model_to_gcs(bucket_name, model_name, model_path)
        logger.info("Model saved successfully")

    @handle_exceptions
    def load_model(self, bucket_name, model_name):
        from .gcs_utils import load_model_from_gcs
        logger.info(f"Loading model from bucket: {bucket_name}, model name: {model_name}")
        self.model = load_model_from_gcs(bucket_name, model_name)
        self.encoder = models.Model(inputs=self.model.input, outputs=self.model.layers[-4].output)
        logger.info("Model loaded successfully")

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