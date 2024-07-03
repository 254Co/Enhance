from statsmodels.tsa.arima.model import ARIMA
from pyspark.sql import DataFrame
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_arima_analysis(df: DataFrame, target_col: str, order: tuple = (1,1,1)):
    logger.info("Running ARIMA analysis")
    try:
        pandas_df = df.toPandas()
        model = ARIMA(pandas_df[target_col], order=order)
        fit_model = model.fit()
        logger.info("ARIMA analysis completed successfully")
        return fit_model
    except Exception as e:
        logger.error(f"Error during ARIMA analysis: {str(e)}", exc_info=True)
        raise e