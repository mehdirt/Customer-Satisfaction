import logging

import mlflow
import pandas as pd
from sklearn.base import RegressorMixin
from zenml import step
from zenml.client import Client

from src.model_dev import LinearRegressionModel
from .config import ModelNameConfig

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def train_model(X_train, X_test, y_train, y_test, config: ModelNameConfig) -> RegressorMixin:
    """
    Train the model on the ingested data.
    Args:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    config: ModelNameConfig
    
    Returns:
        RegressorMixin
    """
    try:
        model = None
        if config.model_name == "LinearRegression":
            mlflow.sklearn.autolog()
            model = LinearRegressionModel()
            trained_model = LinearRegressionModel.train(self='', X_train=X_train, y_train=y_train) #! TODO: 'self' must be removed
            return trained_model
        else:
            raise ValueError(f"Model {config.model_name} is not supported.")
    except Exception as err:
        logging.error(f"Error occured in training the model: {err}")
        raise err

