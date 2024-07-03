import pytest
from pyspark.sql import SparkSession
from data_processing.data_transformation.advanced_data_cleaning import AdvancedDataCleaner
from data_processing.data_transformation.data_imputer import DataImputer
from data_processing.data_transformation.outlier_remover import OutlierRemover
from data_processing.data_transformation.categorical_encoder import CategoricalEncoder
from data_processing.data_transformation.datetime_feature_extractor import DatetimeFeatureExtractor

@pytest.fixture(scope="module")
def spark():
    return SparkSession.builder.appName("test").getOrCreate()

def test_impute_missing_values(spark):
    df = spark.createDataFrame([(1, None), (2, 2.0), (None, 3.0)], ["col1", "col2"])
    result = AdvancedDataCleaner.impute_missing_values(df, columns=['col1'], strategy='mean')
    result_data = result.collect()

    assert result_data[0]['col1'] == 1
    assert result_data[2]['col1'] == 1.5  # assuming 1.5 is the mean

def test_remove_outliers(spark):
    df = spark.createDataFrame([(1, 100), (2, 200), (3, 1000)], ["col1", "col2"])
    result = AdvancedDataCleaner.remove_outliers(df, columns=['col2'], lower_bound=0.05, upper_bound=0.95)
    result_data = result.collect()

    assert len(result_data) == 2  # the outlier row should be removed

def test_impute_with_mode(spark):
    df = spark.createDataFrame([(1, 'A'), (2, 'B'), (3, 'A'), (4, None)], ["col1", "col2"])
    result = DataImputer.impute_with_mode(df, columns=['col2'])
    result_data = result.collect()

    assert result_data[3]['col2'] == 'A'  # mode is 'A'

def test_z_score_method(spark):
    df = spark.createDataFrame([(1, 100), (2, 200), (3, 1000)], ["col1", "col2"])
    result = OutlierRemover.z_score_method(df, columns=['col2'], threshold=2.0)
    result_data = result.collect()

    assert len(result_data) == 2  # the outlier row should be removed

def test_one_hot_encode(spark):
    df = spark.createDataFrame([(1, 'A'), (2, 'B'), (3, 'A')], ["col1", "col2"])
    result = CategoricalEncoder.one_hot_encode(df, columns=['col2'])
    result_data = result.collect()

    assert 'col2_onehot' in result.columns

def test_extract_datetime_features(spark):
    df = spark.createDataFrame([(1, "2021-01-01 12:00:00"), (2, "2021-01-02 13:30:00")], ["col1", "col2"])
    df = df.withColumn("col2", df["col2"].cast("timestamp"))
    result = DatetimeFeatureExtractor.extract_features(df, column='col2')
    result_data = result.collect()

    assert 'col2_year' in result.columns
    assert result_data[0]['col2_year'] == 2021