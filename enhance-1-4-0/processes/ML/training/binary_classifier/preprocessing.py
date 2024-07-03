"""
Data preprocessing module.

Author: 254StudioZ LLC
Date: 2024-07-01
"""

from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, when

def preprocess_data(data: DataFrame, label_col: str) -> DataFrame:
    """
    Preprocesses the given data by assembling features and scaling them.

    Parameters:
    data (DataFrame): The input data.
    label_col (str): The name of the column to be used as the label.

    Returns:
    DataFrame: The preprocessed data with scaled features.
    """
    # Ensure the label column exists
    if label_col not in data.columns:
        raise ValueError(f"Label column '{label_col}' does not exist in the dataset.")

    # Selecting only numeric columns for features
    feature_columns = [column for column, dtype in data.dtypes if dtype in ['int', 'double'] and column != label_col]
    
    # Check if the feature_columns list is empty
    if not feature_columns:
        raise ValueError("No numeric columns found in the dataset.")

    assembler = VectorAssembler(inputCols=feature_columns, outputCol='features')
    assembled_data = assembler.transform(data)
    
    scaler = StandardScaler(inputCol='features', outputCol='scaledFeatures')
    scaled_data = scaler.fit(assembled_data).transform(assembled_data)
    
    # Convert label column to binary integer labels
    scaled_data = scaled_data.withColumn("label", when(col(label_col) > 0, 1).otherwise(0).cast("int"))
    
    return scaled_data
