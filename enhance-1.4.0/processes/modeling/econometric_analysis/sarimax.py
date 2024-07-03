from statsmodels.tsa.statespace.sarimax import SARIMAX
from pyspark.sql import DataFrame
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_sarimax_analysis(df: DataFrame, target_col: str, order: tuple = (1, 1, 1), seasonal_order: tuple = (1, 1, 1, 12)):
    logger.info("Running SARIMAX analysis")
    try:
        pandas_df = df.toPandas()
        model = SARIMAX(pandas_df[target_col], order=order, seasonal_order=seasonal_order)
        fit_model = model.fit()
        logger.info("SARIMAX analysis completed successfully")
        return fit_model
    except Exception as e:
        logger.error(f"Error during SARIMAX analysis: {str(e)}", exc_info=True)
        raise e