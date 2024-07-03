# time_series_analysis/model.py

import tensorflow as tf

class RNNModel:
    def __init__(self, config):
        self.config = config
        self.model = self.create_model()

    def create_model(self):
        model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(self.config.SEQUENCE_LENGTH, 1)),
            tf.keras.layers.LSTM(50),
            tf.keras.layers.Dense(1)
        ])
        optimizer = tf.keras.optimizers.Adam(self.config.LEARNING_RATE)
        model.compile(optimizer=optimizer, loss='mean_squared_error')
        return model
