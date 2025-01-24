import logging
import pandas as pd
from zenml import step

class IngestData:
    """
    A class responsible for ingesting data from a specified file path.

    Attributes:
        data_path (str): The path to the data file to be ingested.
    """
    def __init__(self, data_path: str) -> None:
        """
        Initializes the IngestData class with the specified data path.

        Args:
            data_path (str): The path to the data file to be ingested.
        """
        self.data_path = data_path

    def get_data(self):
        """
        Reads data from the specified file path and returns it as a Pandas DataFrame.

        Returns:
            pd.DataFrame: The ingested data as a Pandas DataFrame.

        Raises:
            FileNotFoundError: If the specified file path does not exist.
            pd.errors.ParserError: If there is an error while parsing the CSV file.
        """
        logging.info(f"Ingesting Data from {self.data_path}")
        return pd.read_csv(self.data_path)
    
@step
def ingest_data(data_path: str) -> pd.DataFrame:
    """
    A ZenML step to ingest data from a specified file path.

    This function serves as a wrapper for the `IngestData` class, handling potential exceptions during the data ingestion process.

    Args:
        data_path (str): The path to the data file to be ingested.

    Returns:
        pd.DataFrame: The ingested data as a Pandas DataFrame.

    Raises:
        Exception: If there is an error during the data ingestion process.
    """
    try: 
        ingest_data = IngestData(data_path)
        dframe = ingest_data.get_data()
        return dframe
    except Exception as err:
        logging.error(f"Error while ingesting data: {err}")
        raise err