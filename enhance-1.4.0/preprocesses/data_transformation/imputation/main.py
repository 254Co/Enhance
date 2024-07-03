# main.py

from pyspark.sql import SparkSession
from imputation import (
    SparkMeanImputer, SparkMedianImputer, SparkModeImputer, 
    SparkFFillImputer, SparkBFillImputer, SparkKNNImputer, 
    SparkMICEImputer, SparkInterpolationImputer, SparkRegressionImputer
)

def main():
    spark = SparkSession.builder.appName("ImputationExample").getOrCreate()

    # Sample data
    data = [(1, 2, None, 4, 5), (None, 2, 3, None, 5), (1, None, None, 4, 5)]
    columns = ["A", "B", "C", "D", "E"]
    df = spark.createDataFrame(data, columns)

    # Mean Imputation
    df_mean = SparkMeanImputer.impute(df, ["A", "B", "C", "D", "E"])
    df_mean.show()

    # Median Imputation
    df_median = SparkMedianImputer.impute(df, ["A", "B", "C", "D", "E"])
    df_median.show()

    # Mode Imputation
    df_mode = SparkModeImputer.impute(df, ["A", "B", "C", "D", "E"])
    df_mode.show()

    # Forward Fill Imputation
    df_ffill = SparkFFillImputer.impute(df, ["A", "B", "C", "D", "E"])
    df_ffill.show()

    # Backward Fill Imputation
    df_bfill = SparkBFillImputer.impute(df, ["A", "B", "C", "D", "E"])
    df_bfill.show()

    # KNN Imputation
    df_knn = SparkKNNImputer.impute(df, ["A", "B", "C", "D", "E"])
    df_knn.show()

    # MICE Imputation
    df_mice = SparkMICEImputer.impute(df, ["A", "B", "C", "D", "E"])
    df_mice.show()

    # Interpolation Imputation
    df_interpolation = SparkInterpolationImputer.impute(df, ["A", "B", "C", "D", "E"], method='linear')
    df_interpolation.show()

    # Regression Imputation
    df_regression = SparkRegressionImputer.impute(df, "A", ["B", "C", "D", "E"])
    df_regression.show()

    spark.stop()

if __name__ == "__main__":
    main()
