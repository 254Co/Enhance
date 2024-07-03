from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import logging

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def build_lstm_model(input_shape):
    """
    Build and compile an LSTM model.

    Args:
        input_shape (tuple): Shape of the input data.

    Returns:
        model (Sequential): Compiled LSTM model.
    """
    try:
        logger.info("Building LSTM model")
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
        model.add(LSTM(50))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        logger.info("LSTM model built and compiled successfully")
        return model
    except Exception as e:
        logger.error(f"Error building LSTM model: {str(e)}", exc_info=True)
        raise e