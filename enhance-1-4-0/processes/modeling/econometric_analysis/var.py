from statsmodels.tsa.api import VAR
from pyspark.sql import DataFrame
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_vector_autoregression(df: DataFrame, target_cols: list, maxlags: int = 1):
    logger.info("Running Vector Autoregression (VAR) analysis")
    try:
        pandas_df = df.select(target_cols).toPandas()
        model = VAR(pandas_df)
        fit_model = model.fit(maxlags=maxlags)
        logger.info("VAR analysis completed successfully")
        return fit_model
    except Exception as e:
        logger.error(f"Error during VAR analysis: {str(e)}", exc_info=True)
        raise e