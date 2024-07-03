from causalimpact import CausalImpact
from pyspark.sql import DataFrame
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_causal_impact_analysis(df: DataFrame, pre_period: list, post_period: list, target_col: str):
    logger.info("Running Causal Impact analysis")
    try:
        pandas_df = df.toPandas()
        ci = CausalImpact(pandas_df[target_col], pre_period, post_period)
        logger.info("Causal Impact analysis completed successfully")
        return ci
    except Exception as e:
        logger.error(f"Error during Causal Impact analysis: {str(e)}", exc_info=True)
        raise e