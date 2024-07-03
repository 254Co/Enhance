import h2o
from h2o.automl import H2OAutoML
from pyspark.sql import DataFrame
import logging

h2o.init()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_automl(df: DataFrame, features: list, label: str, max_runtime_secs: int = 3600):
    logger.info("Running AutoML")
    try:
        h2o_df = h2o.H2OFrame(df.toPandas())
        x = features
        y = label

        aml = H2OAutoML(max_runtime_secs=max_runtime_secs)
        aml.train(x=x, y=y, training_frame=h2o_df)

        logger.info("AutoML run completed successfully")
        return aml.leader
    except Exception as e:
        logger.error(f"Error during AutoML run: {str(e)}", exc_info=True)
        raise e