from sklearn.linear_model import Lasso
from pyspark.sql import DataFrame
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_lasso_regression(df: DataFrame, features: list, label: str, alpha: float = 1.0):
    logger.info("Running Lasso regression analysis")
    try:
        # Assemble features into a feature vector
        assembler = VectorAssembler(inputCols=features, outputCol="features")
        df_prepared = assembler.transform(df)

        # Initialize Lasso model
        lasso = LinearRegression(featuresCol="features", labelCol=label, elasticNetParam=1.0, regParam=alpha)

        # Set up cross-validation and parameter grid
        paramGrid = ParamGridBuilder() \
            .addGrid(lasso.regParam, [0.01, 0.1, 1.0]) \
            .build()

        crossval = CrossValidator(
            estimator=lasso,
            estimatorParamMaps=paramGrid,
            evaluator=RegressionEvaluator(labelCol=label, predictionCol="prediction", metricName="rmse"),
            numFolds=3,
            parallelism=4  # Utilize parallelism for cross-validation
        )

        # Fit the model
        cvModel = crossval.fit(df_prepared)

        logger.info("Lasso regression analysis completed successfully")
        return cvModel
    except Exception as e:
        logger.error(f"Error during Lasso regression analysis: {str(e)}", exc_info=True)
        raise e