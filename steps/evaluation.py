import logging
from typing import Tuple
from typing_extensions import Annotated

import pandas as pd
from sklearn.base import RegressorMixin
from zenml import step
from src.evaluation import MSE, RMSE, R2

@step
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

        # Calculate RMSE
        rmse_obj= RMSE()
        rmse = rmse_obj.calculate_scores(y_test, prediction)
        
        # Calculate R2 score
        r2_obj= R2()
        r2_score = r2_obj.calculate_scores(y_test, prediction)
        
        return r2_score, rmse
    except Exception as err:
        logging.error(f"Error occured in evaluating the model: {err}")
        raise err