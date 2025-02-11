import logging
from abc import ABC, abstractmethod

import numpy as np
from sklearn.metrics import mean_squared_error, root_mean_squared_error, r2_score

class Evalueation(ABC):
    """
    Abstract class defining strategy for evaluating our models
    """
    @abstractmethod
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate the scores for the model.
        Args:
            y_true: true labels
            y_pred: predicted labels
        
        Returns:
            float
        """
        pass

class MSE(Evalueation):
    """
    Evaluation strategy that uses Mean Squared Error (MSE).
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate the Mean Squared Error (MSE).
        Args:
            y_true: true labels
            y_pred: predicted labels
        
        Returns:
            float
        """
        try:
            logging.info("Calculating MSE")
            mse = mean_squared_error(y_true, y_pred)
            logging.info(f"MSE: {mse}")
            return mse
        except Exception as err:
            logging.error(f"Error occured in calculating MSE: {err}")
            raise err

class R2(Evalueation):
    """Evaluation Strategy that uses R2 score."""
    def calculate_scores(self, y_true, y_pred) -> float:
        """
        Calculate the R2 score.
        Args:
            y_true: true labels
            y_pred: predicted labels
        
        Returns:
            float
        """
        try:
            logging.info("Calculating R2 Score")
            r2 = r2_score(y_true, y_pred)
            logging.info(f"R2: {r2}")
            return r2
        except Exception as err:
            logging.error(f"Error occured in calculating MSE: {err}")
            raise err

class RMSE(Evalueation):
    """Evaluation Strategy that uses Root Mean Squared Error (RMSE) score."""
    def calculate_scores(self, y_true, y_pred) -> float:
        """
        Calculate the RMSE score.
        Args:
            y_true: true labels
            y_pred: predicted labels
        
        Returns:
            float
        """
        try:
            logging.info("Calculating R2 Score")
            rmse = root_mean_squared_error(y_true, y_pred)
            logging.info(f"RMSE: {rmse}")
            return rmse
        except Exception as err:
            logging.error(f"Error occured in calculating MSE: {err}")
            raise err