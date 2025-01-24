from zenml import pipeline
from steps.ingest_data import ingest_dframe
from steps.clean_data import clean_dframe
from steps.model_train import train_model
from steps.evaluation import evaluate_model

@pipeline
def train_pipeline(data_path: str):
    """
    """
    df = ingest_dframe(data_path)
    df = clean_dframe(df)
    train_model(df)
    evaluate_model(df)


