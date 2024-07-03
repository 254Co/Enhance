import logging
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, GBTClassifier, RandomForestClassifier, NaiveBayes
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from enhance.Enhance.utils.logger import get_logger
from enhance.Enhance.utils.exception_handler import handle_exceptions
from enhance.Enhance.data_processing.data_transformation.utils import ParallelProcessingUtils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@handle_exceptions
def run_federated_learning(df: DataFrame, features: list, label: str):
    logger.info("Running federated learning")
    try:
        spark = SparkSession.builder.getOrCreate()
        num_partitions = df.rdd.getNumPartitions()
        logger.info(f"DataFrame has {num_partitions} partitions")

        def train_model_on_partition(iterator):
            local_df = spark.createDataFrame(iterator, df.schema)
            assembler = VectorAssembler(inputCols=features, outputCol="features")
            local_df = assembler.transform(local_df)
            
            models = {
                "logistic_regression": LogisticRegression(labelCol=label, featuresCol="features"),
                "decision_tree": DecisionTreeClassifier(labelCol=label, featuresCol="features"),
                "gbt": GBTClassifier(labelCol=label, featuresCol="features"),
                "random_forest": RandomForestClassifier(labelCol=label, featuresCol="features"),
                "naive_bayes": NaiveBayes(labelCol=label, featuresCol="features")
            }
            
            results = {}
            for model_name, model in models.items():
                pipeline = Pipeline(stages=[model])
                param_grid = ParamGridBuilder().build()
                cross_validator = CrossValidator(
                    estimator=pipeline,
                    estimatorParamMaps=param_grid,
                    evaluator=BinaryClassificationEvaluator(labelCol=label),
                    numFolds=3,
                    parallelism=4
                )
                cv_model = cross_validator.fit(local_df)
                results[model_name] = cv_model.bestModel
                logger.info(f"Trained {model_name} model on partition")
            
            return [results]

        parallel_utils = ParallelProcessingUtils()
        model_results = parallel_utils.parallelize_dataframe_processing(df, train_model_on_partition, num_partitions)
        final_models = model_results.collect()
        
        logger.info("Federated learning setup completed successfully")
        return final_models
    except Exception as e:
        logger.error(f"Error during federated learning setup: {str(e)}", exc_info=True)
        raise e