# Initializer for models module
from .automl import run_automl
from .deep_learning import run_deep_learning
from .federated_learning import run_federated_learning

__all__ = [
    'run_automl',
    'run_deep_learning',
    'run_federated_learning'
]