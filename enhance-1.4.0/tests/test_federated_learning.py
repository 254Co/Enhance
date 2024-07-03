import pytest
from pyspark.sql import SparkSession
from processes.models.federated_learning import run_federated_learning

@pytest.fixture(scope="module")
def spark():
    return SparkSession.builder.appName("test").getOrCreate()

def test_run_federated_learning(spark):
    df = spark.createDataFrame([
        (0, 1.0, 3.0, 0),
        (1, 2.0, 2.0, 1),
        (0, 0.0, 1.0, 2)
    ], ["label", "feature1", "feature2", "feature3"])

    features = ["feature1", "feature2", "feature3"]
    label = "label"

    model = run_federated_learning(df, features, label)
    assert model is not None