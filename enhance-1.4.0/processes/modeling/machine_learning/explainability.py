import shap
from pyspark.sql import DataFrame
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def explain_model_predictions(model, df: DataFrame):
    logger.info("Explaining model predictions")
    explainer = shap.Explainer(model)
    shap_values = explainer(df.toPandas())
    shap.summary_plot(shap_values, df.toPandas())
