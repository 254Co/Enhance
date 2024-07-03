from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark.sql import SparkSession
from enhance.Enhance.utils.logger import get_logger
from enhance.Enhance.utils.exception_handler import handle_exceptions

logger = get_logger(__name__)

class ParallelProcessingUtils:

    @staticmethod
    @handle_exceptions
    def parallelize_dataframe_processing(df: DataFrame, processing_func, num_partitions: int = None) -> DataFrame:
        """
        Parallelize the processing of DataFrame by repartitioning and applying a processing function.
        """
        num_partitions = num_partitions or df.rdd.getNumPartitions()
        logger.info(f"Parallelizing DataFrame processing with {num_partitions} partitions")
        df = df.repartition(num_partitions)

        def process_partition(iterator):
            spark = SparkSession.builder.getOrCreate()
            batch = list(iterator)
            batch_df = spark.createDataFrame(batch)
            result = processing_func(batch_df)
            return result.collect()

        processed_rdd = df.rdd.mapPartitions(process_partition)
        result_df = processed_rdd.toDF()

        return result_df