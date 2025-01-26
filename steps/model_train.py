import logging
import pandas as pd
from zenml import step
from pydantic import BaseModel

# Create a Pydantic model with arbitrary_types_allowed
class DataFrameConfig(BaseModel):
    df: pd.DataFrame

    class Config:
        arbitrary_types_allowed = True

@step
def train_model(dframe: DataFrameConfig) -> None:
    """
    """
    pass