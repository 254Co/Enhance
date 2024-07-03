from pyspark.sql import SparkSession
from processes.ML.DL.RNNs_LSTMs.sequential_data_analysis.training import Trainer
from processes.ML.DL.RNNs_LSTMs.sequential_data_analysis.config import Config
from processes.ML.DL.RNNs_LSTMs.sequential_data_analysis.data_loader import DataLoader
from processes.ML.DL.RNNs_LSTMs.sequential_data_analysis.preprocessing import Preprocessor

def main():
    spark = SparkSession.builder \
        .appName("SequentialDataAnalysisWithRNN") \
        .config("spark.executor.memory", "4g") \
        .config("spark.driver.memory", "4g") \
        .config("spark.sql.shuffle.partitions", "200") \
        .config("spark.dynamicAllocation.enabled", "true") \
        .config("spark.dynamicAllocation.minExecutors", "1") \
        .config("spark.dynamicAllocation.maxExecutors", "10") \
        .getOrCreate()

    config = Config()
    data_loader = DataLoader()
    data_df = data_loader.load_data("path/to/data.csv")

    preprocessor = Preprocessor(config)
    processed_data_df = preprocessor.preprocess_data(data_df, "data_column")

    trainer = Trainer(model=None, config=config)
    trainer.train(processed_data_df)

    spark.stop()

if __name__ == "__main__":
    main()