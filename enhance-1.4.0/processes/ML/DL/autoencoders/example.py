from pyspark_autoencoders import train_autoencoder

# Start the training process with continuous data streaming from Polygon.io
autoencoder = train_autoencoder(input_dim=30, bucket_name='your_bucket_name', model_name='autoencoder_model', polygon_api_key='your_polygon_api_key', batch_size=256)

# Dimensionality reduction
reduced_data = autoencoder.dimensionality_reduction(data)

# Anomaly detection
anomalies = autoencoder.anomaly_detection(data, threshold=0.1)

# Noise reduction
clean_data = autoencoder.noise_reduction(noisy_data)

# Save the model to GCS
autoencoder.save_model(bucket_name='your_bucket_name', model_name='autoencoder_model')

# Load the model from GCS
autoencoder.load_model(bucket_name='your_bucket_name', model_name='autoencoder_model')
