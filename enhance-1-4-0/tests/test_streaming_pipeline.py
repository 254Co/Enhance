import pytest
from unittest.mock import patch
from processes.streaming.streaming_pipeline import StreamingPipeline

@pytest.fixture(scope="module")
def spark():
    from pyspark.sql import SparkSession
    return SparkSession.builder.appName("test").getOrCreate()

def test_start_streaming(spark):
    with patch("processes.streaming.streaming_pipeline.SparkSession.builder.getOrCreate") as mock_spark_session, \
         patch("processes.streaming.streaming_pipeline.SparkSession.readStream") as mock_read_stream, \
         patch.object(StreamingPipeline, 'start_streaming', return_value=None) as mock_start_streaming:

        StreamingPipeline.start_streaming("input_topic", "output_topic", "localhost:9092")
        mock_start_streaming.assert_called_once_with("input_topic", "output_topic", "localhost:9092")