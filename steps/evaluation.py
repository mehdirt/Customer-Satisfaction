import logging
from typing import Tuple
from typing_extensions import Annotated

import mlflow
import pandas as pd
from sklearn.base import RegressorMixin
from zenml import step
from zenml.client import Client
from src.evaluation import MSE, RMSE, R2

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def evaluate_model(
    model: RegressorMixin,
    X_test: pd.DataFrame,
    y_test: pd.Series
    ) -> Tuple[
        Annotated[float, "r2_score"],
        Annotated[float, "rmse"]
    ]:
    """
    Evaluate the model on the ingested data.
    Args:
        model: RegressorMinxin
        X_test: pd.DataFrame
        y_test: pd.Series
    Returns:
        Tuple[float, float]
    """
    try:
        prediction = model.predict(X_test)
        # Calculate MSE
        mse_obj= MSE()
        mse = mse_obj.calculate_scores(y_test, prediction)
        mlflow.log_metric("mse", mse)

        # Calculate RMSE
        rmse_obj= RMSE()
        rmse = rmse_obj.calculate_scores(y_test, prediction)
        mlflow.log_metric("rmse", rmse)

        # Calculate R2 score
        r2_obj= R2()
        r2_score = r2_obj.calculate_scores(y_test, prediction)
        mlflow.log_metric("r2", r2_score)

        return r2_score, rmse
    except Exception as err:
        logging.error(f"Error occured in evaluating the model: {err}")
        raise err