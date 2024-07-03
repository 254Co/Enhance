from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import DataFrame
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_regression_analysis(df: DataFrame, features: list, label: str) -> LinearRegression:
    logger.info("Running linear regression analysis")
    try:
        assembler = VectorAssembler(inputCols=features, outputCol='features')
        df_prepared = assembler.transform(df)

        lr = LinearRegression(featuresCol='features', labelCol=label)

        logger.info("Creating parameter grid for cross-validation")
        paramGrid = ParamGridBuilder() \
            .addGrid(lr.regParam, [0.01, 0.1, 1.0]) \
            .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]) \
            .build()

        logger.info("Setting up cross-validator")
        crossval = CrossValidator(
            estimator=lr,
            estimatorParamMaps=paramGrid,
            evaluator=RegressionEvaluator(labelCol=label, predictionCol="prediction", metricName="rmse"),
            numFolds=3,
            parallelism=4  # Utilize parallelism for cross-validation
        )

        logger.info("Fitting cross-validator to prepared data")
        cvModel = crossval.fit(df_prepared)

        logger.info("Linear regression analysis completed successfully")
        return cvModel
    except Exception as e:
        logger.error(f"Error during linear regression analysis: {str(e)}", exc_info=True)
        raise e