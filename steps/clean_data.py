import logging
import pandas as pd
from zenml import step

from src.data_cleaning import DataCleaning, DataPreProcessStrategy, DataSplitStrategy
from typing_extensions import Annotated
from typing import Tuple

@step
def clean_dframe(dframe: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],
]: 
    """
    Cleans and preprocesses the input dataframe, then splits it into training and testing datasets.

    This function performs the following tasks:
    1. Preprocesses the input dataframe by handling missing values, encoding categorical features,
       and applying other data cleaning strategies.
    2. Splits the cleaned dataframe into features (X_train, X_test) and target labels (y_train, y_test)
       for training and testing the machine learning model.

    Args:
        dframe (pd.DataFrame): The input data to be cleaned and split.

    Returns:
        Tuple[Annotated[pd.DataFrame, "X_train"], Annotated[pd.DataFrame, "X_test"], 
               Annotated[pd.Series, "y_train"], Annotated[pd.Series, "y_test"]]:
            - X_train: The training features (cleaned).
            - X_test: The testing features (cleaned).
            - y_train: The training target labels.
            - y_test: The testing target labels.

    Raises:
        Exception: If there is an error during the data cleaning or splitting process.

    Example:
        X_train, X_test, y_train, y_test = clean_dframe(my_dataframe)

    """
    try:
        # Preprocess the data
        process_strategy = DataPreProcessStrategy()
        data_cleaning = DataCleaning(dframe, strategy=process_strategy)
        processed_data = data_cleaning.handle_data()
        logging.info("Data preprocessing finished")

        # Split the data into train/test
        split_strategy = DataSplitStrategy()
        data_cleaning = DataCleaning(dframe, strategy=DataSplitStrategy)
        X_train, X_test, y_train, y_test = data_cleaning.handle_data()
        logging.info("Data splitting finished")
        logging.info("Data Cleaning Completed")
        
        return X_train, X_test, y_train, y_test
    except Exception as err:
        logging.error(f"Error occuered in cleaning data: {err}")
        raise err

    