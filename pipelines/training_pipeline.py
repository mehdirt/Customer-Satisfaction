from zenml import pipeline
from steps.config import ModelNameConfig
from steps.ingest_data import ingest_dframe
from steps.clean_data import clean_dframe
from steps.model_train import train_model
from steps.evaluation import evaluate_model

@pipeline(enable_cache=False)
def train_pipeline(data_path: str):
    """
    """
    df = ingest_dframe(data_path)
    X_train, X_test, y_train, y_test = clean_dframe(df)
    
    model_config = ModelNameConfig()
    model_config.model_name = "LinearRegression"
    
    model = train_model(X_train, X_test, y_train, y_test, model_config)
    
    r2_score, rmse = evaluate_model(model, X_test, y_test)


