# time_series_analysis/config.py

class Config:
    SEQUENCE_LENGTH = 50
    BATCH_SIZE = 32
    EPOCHS = 10
    LEARNING_RATE = 0.001
    MODEL_PATH = 'saved_model/my_model'
    CLEANING_PARAMS = {
        'remove_nulls': [],
        'fill_nulls': {}
    }
    FEATURE_PARAMS = {}
    TRANSFORMATION_PARAMS = {
        'input_cols': [],
        'output_col': 'normalized_features',
        'method': 'min-max'
    }