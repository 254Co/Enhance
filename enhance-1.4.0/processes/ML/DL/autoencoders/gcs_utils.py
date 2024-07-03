# gcs_utils.py
from google.cloud import storage
import tensorflow as tf
import tempfile

def save_model_to_gcs(bucket_name, model_name, model_path):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(model_name)
    blob.upload_from_filename(model_path)

def load_model_from_gcs(bucket_name, model_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(model_name)
    with tempfile.NamedTemporaryFile() as temp_model_file:
        blob.download_to_filename(temp_model_file.name)
        return tf.keras.models.load_model(temp_model_file.name)
