from arch import arch_model
from pyspark.sql import DataFrame
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_garch_analysis(df: DataFrame, target_col: str):
    logger.info("Running GARCH analysis")
    try:
        pandas_df = df.toPandas()
        model = arch_model(pandas_df[target_col], vol='Garch', p=1, q=1)
        fit_model = model.fit()
        logger.info("GARCH analysis completed successfully")
        return fit_model
    except Exception as e:
        logger.error(f"Error during GARCH analysis: {str(e)}", exc_info=True)
        raise e