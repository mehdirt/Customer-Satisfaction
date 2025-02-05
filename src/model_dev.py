import logging
from abc import ABC, abstractmethod

from sklearn.linear_model import LinearRegression

class Model(ABC):
    """
    Abstract class for all models.
    """
    @abstractmethod
    def train(self, X_train, y_train):
        """
        Training the model.
        Args:
            X_train: training data features
            y_train: training labels
        
        Returns:
            None
        """
        pass

class LinearRegressionModel(Model):
    """
    Linear Regression model.
    """
    def train(self, X_train, y_train, **kwargs):
        """
        Training the model.
        Args:
            X_train: training data features
            y_train: training labels
        
        Returns:
            LinearRegression
        """
        try:
            reg = LinearRegression(**kwargs)
            reg.fit(X_train, y_train)
            logging.info("Model training completed")
            return reg
        except Exception as err:
            logging.error(f"Error occured in training the model: {err}")
            raise err