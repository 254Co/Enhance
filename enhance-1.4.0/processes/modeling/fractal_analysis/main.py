# main.py
from pyspark.sql import SparkSession
from fractal_analysis import load_data, advanced_fractal_analysis, build_ml_model, extract_intelligence, save_intelligence

def main():
    spark = SparkSession.builder \
        .appName("AdvancedFractalAnalysis") \
        .getOrCreate()
    
    # Load data from a CSV file (replace with actual file path)
    file_path = "path/to/your/market_data.csv"
    market_data = load_data(spark, file_path)
    
    # Perform advanced fractal analysis
    window_sizes = [30, 60, 90]  # Example window sizes
    fractal_result = advanced_fractal_analysis(market_data, window_sizes, spark)
    
    # Build and evaluate the machine learning model
    ml_result = build_ml_model(fractal_result)
    
    # Print the evaluation metrics
    print(f"RMSE: {ml_result['rmse']}")
    
    # Extract actionable intelligence
    intelligence_insights = extract_intelligence(fractal_result, spark)
    
    # Save the intelligence insights to a specified path
    output_path = "path/to/save/intelligence"
    save_intelligence(intelligence_insights, output_path)
    
    # Stop the Spark session
    spark.stop()

if __name__ == "__main__":
    main()
