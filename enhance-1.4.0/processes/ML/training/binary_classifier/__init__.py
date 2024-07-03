"""
Machine Learning Model Training Package

This package provides modules for training, tuning, and evaluating various machine learning models using Spark MLlib.

Modules:
- decision_tree: Decision Tree training and tuning
- random_forest: Random Forest training and tuning
- svm: Support Vector Machine (SVM) training and tuning
- logistic_regression: Logistic Regression training and tuning
- naive_bayes: Naive Bayes training and tuning
- gbt: Gradient-Boosted Trees training and tuning
- preprocessing: Data preprocessing
- evaluation: Model evaluation
- utils: Utility functions
- main: Main module to orchestrate the process
"""

from .decision_tree import train_decision_tree, tune_decision_tree
from .random_forest import train_random_forest, tune_random_forest
from .svm import train_svm, tune_svm
from .logistic_regression import train_logistic_regression, tune_logistic_regression
from .naive_bayes import train_naive_bayes, tune_naive_bayes
from .gbt import train_gbt, tune_gbt
from .preprocessing import preprocess_data
from .evaluation import evaluate_model
from .utils import get_logger, set_log_level
from .main import main

