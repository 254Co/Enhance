import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, TimestampType
from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StreamingPipeline:
    @staticmethod
    def start_streaming(input_kafka_topic=config['kafka']['input_topic'],
                        output_kafka_topic=config['kafka']['output_topic'],
                        kafka_bootstrap_servers=config['kafka']['bootstrap_servers']):
        """
        Starts the Spark Streaming pipeline to process real-time data from Kafka.

        Args:
        - input_kafka_topic: str, the name of the input Kafka topic
        - output_kafka_topic: str, the name of the output Kafka topic
        - kafka_bootstrap_servers: str, the Kafka bootstrap servers
        """
        try:
            spark = SparkSession.builder \
                .appName("RealTimeDataProcessing") \
                .getOrCreate()

            logger.info("Spark session started for streaming")

            # Define schema for incoming data
            schema = StructType([
                StructField("id", StringType(), True),
                StructField("value", DoubleType(), True),
                StructField("timestamp", TimestampType(), True)
            ])

            # Read data from Kafka
            raw_data = spark.readStream \
                .format("kafka") \
                .option("kafka.bootstrap.servers", kafka_bootstrap_servers) \
                .option("subscribe", input_kafka_topic) \
                .option("startingOffsets", "latest") \
                .load()

            logger.info(f"Subscribed to Kafka topic: {input_kafka_topic}")

            # Extract the value column and convert to string
            json_data = raw_data.selectExpr("CAST(value AS STRING)")

            # Parse the JSON data
            parsed_data = json_data.select(from_json(col("value"), schema).alias("data")).select("data.*")

            logger.info("Parsed JSON data from Kafka stream")

            # Example Transformation: Simple Moving Average
            transformed_data = parsed_data \
                .groupBy(col("id")) \
                .agg({"value": "avg"})

            logger.info("Applied transformation to data")

            # Serialize the data to JSON
            output_data = transformed_data.selectExpr("CAST(id AS STRING) AS key", "to_json(struct(*)) AS value")

            # Write data back to Kafka
            query = output_data.writeStream \
                .format("kafka") \
                .option("kafka.bootstrap.servers", kafka_bootstrap_servers) \
                .option("topic", output_kafka_topic) \
                .option("checkpointLocation", "/tmp/checkpoints") \
                .outputMode("complete") \
                .start()

            logger.info(f"Writing transformed data to Kafka topic: {output_kafka_topic}")

            query.awaitTermination()

        except Exception as e:
            logger.error(f"Error in streaming pipeline: {str(e)}", exc_info=True)
            raise