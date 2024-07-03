"""
Main module to orchestrate the data processing, model training, tuning, and evaluation.

Author: 254StudioZ LLC
Date: 2024-07-01
"""

from pyspark.sql import SparkSession
import pandas as pd
import yfinance as yf  

from utils import get_logger
from preprocessing import preprocess_data
from decision_tree import train_decision_tree, tune_decision_tree
from random_forest import train_random_forest, tune_random_forest
from svm import train_svm, tune_svm
from logistic_regression import train_logistic_regression, tune_logistic_regression
from naive_bayes import train_naive_bayes, tune_naive_bayes
from gbt import train_gbt, tune_gbt
from evaluation import evaluate_model

# Function to fetch data from Yahoo Finance
def fetch_data(ticker, period='1y'):
    data = yf.download(ticker, period=period)
    return data

def main(target_ticker, feature_tickers):
    spark = SparkSession.builder.appName("ModelTrainingPipeline").getOrCreate()
    logger = get_logger("ModelTrainingPipeline")

    logger.info("Starting data processing for target ticker...")
    # Fetch and process target data
    target_data = fetch_data(target_ticker, period='max')
    target_data['Target'] = (target_data['Close'].shift(-1) > target_data['Close']).astype(int)
    target_data = target_data[:-1]

    # Fetch and process feature data
    feature_data = []
    for ticker in feature_tickers:
        logger.info(f"Fetching data for feature ticker: {ticker}")
        data = fetch_data(ticker, period='max')
        data = data[['Close']].rename(columns={'Close': f'{ticker}_Close'})
        feature_data.append(data)

    # Combine target data and feature data
    combined_data = target_data.copy()
    for data in feature_data:
        combined_data = combined_data.join(data, how='inner')

    # Convert Pandas DataFrame to Spark DataFrame
    logger.info("Converting data to Spark DataFrame...")
    spark_df = spark.createDataFrame(combined_data.reset_index())

    # Preprocess data
    logger.info("Preprocessing data...")
    processed_data = preprocess_data(spark_df, label_col="Target")

    logger.info("Training and tuning models...")
    dt_model, dt_params = tune_decision_tree(processed_data)
    rf_model, rf_params = tune_random_forest(processed_data)
    svm_model, svm_params = tune_svm(processed_data)
    lr_model, lr_params = tune_logistic_regression(processed_data)
    nb_model, nb_params = tune_naive_bayes(processed_data)
    gbt_model, gbt_params = tune_gbt(processed_data)

    logger.info("Evaluating models...")
    dt_predictions = dt_model.transform(processed_data)
    rf_predictions = rf_model.transform(processed_data)
    svm_predictions = svm_model.transform(processed_data)
    lr_predictions = lr_model.transform(processed_data)
    nb_predictions = nb_model.transform(processed_data)
    gbt_predictions = gbt_model.transform(processed_data)

    dt_metrics = evaluate_model(dt_predictions)
    rf_metrics = evaluate_model(rf_predictions)
    svm_metrics = evaluate_model(svm_predictions)
    lr_metrics = evaluate_model(lr_predictions)
    nb_metrics = evaluate_model(nb_predictions)
    gbt_metrics = evaluate_model(gbt_predictions)

    logger.info(f"Decision Tree metrics: {dt_metrics}")
    logger.info(f"Random Forest metrics: {rf_metrics}")
    logger.info(f"SVM metrics: {svm_metrics}")
    logger.info(f"Logistic Regression metrics: {lr_metrics}")
    logger.info(f"Naive Bayes metrics: {nb_metrics}")
    logger.info(f"GBT metrics: {gbt_metrics}")

    spark.stop()

if __name__ == "__main__":
    target_ticker = 'SPY'  # Example target ticker
    feature_tickers = ['GOOGL', 'MSFT', 'AMZN', 'AAPL', 'T', 'VZ', 'BA', 'CAT', 'PG']  # Example feature tickers
    main(target_ticker, feature_tickers)
