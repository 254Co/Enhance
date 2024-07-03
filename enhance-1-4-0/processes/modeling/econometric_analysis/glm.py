from pyspark.ml.regression import GeneralizedLinearRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import DataFrame
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_glm_analysis(df: DataFrame, features: list, label: str, family: str = 'gaussian') -> GeneralizedLinearRegression:
    logger.info("Running GLM analysis")
    try:
        assembler = VectorAssembler(inputCols=features, outputCol='features')
        df_prepared = assembler.transform(df)

        glm = GeneralizedLinearRegression(featuresCol='features', labelCol=label, family=family)
        glm_model = glm.fit(df_prepared)

        logger.info("GLM analysis completed successfully")
        return glm_model
    except Exception as e:
        logger.error(f"Error during GLM analysis: {str(e)}", exc_info=True)
        raise e