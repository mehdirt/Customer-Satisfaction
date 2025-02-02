import logging

import pandas as pd
from zenml import step
from sklearn.base import RegressorMixin

from src.model_dev import LinearRegressionModel
from .config import ModelNameConfig

@step
def train_model(X_train, X_test, y_train, y_test, config) -> RegressorMixin:
    """
    Train the model on the ingested data.
    Args:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.DataFrame
    y_test: pd.DataFrame
    config: ModelNameConfig
    
    Returns:
        RegressorMixin 
    """
    try:
        model = None
        if config.model_name == "LinearRegression":
            model = LinearRegressionModel()
            trained_model = LinearRegressionModel.train(X_train, y_train)
            return trained_model
        else:
            raise ValueError(f"Model {config.model_name} is not supported.")
    except Exception as err:
        logging.error(f"Error occured in training the model: {err}")
        raise err

