from pyspark.sql import DataFrame
from .binary_classifier.decision_tree import train_decision_tree
from .binary_classifier.logistic_regression import train_logistic_regression
from .binary_classifier.gbt import train_gbt
from .binary_classifier.naive_bayes import train_naive_bayes
from .binary_classifier.random_forest import train_random_forest
from .binary_classifier.svm import train_svm
import logging

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class ModelFactory:
    def __init__(self, algorithm: str, params: dict):
        self.algorithm = algorithm
        self.params = params

    def train_model(self, input_df: DataFrame):
        try:
            logger.info(f"Training model using algorithm: {self.algorithm}")
            if self.algorithm == "decision_tree":
                return train_decision_tree(input_df, **self.params)
            elif self.algorithm == "logistic_regression":
                return train_logistic_regression(input_df, **self.params)
            elif self.algorithm == "gbt":
                return train_gbt(input_df, **self.params)
            elif self.algorithm == "naive_bayes":
                return train_naive_bayes(input_df, **self.params)
            elif self.algorithm == "random_forest":
                return train_random_forest(input_df, **self.params)
            elif self.algorithm == "svm":
                return train_svm(input_df, **self.params)
            else:
                raise ValueError(f"Unknown algorithm: {self.algorithm}")
        except Exception as e:
            logger.error(f"Error during model training: {str(e)}", exc_info=True)
            raise e