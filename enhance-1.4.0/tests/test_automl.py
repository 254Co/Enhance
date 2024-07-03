import pytest
from pyspark.sql import SparkSession
from processes.models.automl import run_automl

@pytest.fixture(scope="module")
def spark():
    return SparkSession.builder.appName("test").getOrCreate()

def test_run_automl(spark):
    df = spark.createDataFrame([
        (0, 1.0, 3.0, 0),
        (1, 2.0, 2.0, 1),
        (0, 0.0, 1.0, 2)
    ], ["label", "feature1", "feature2", "feature3"])

    features = ["feature1", "feature2", "feature3"]
    label = "label"

    model = run_automl(df, features, label)
    assert model is not None