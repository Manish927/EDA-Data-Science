import pandas as pd

from src.preprocessing import preprocess_data, split_data
from src.train_models import train_decision_tree
from src.evaluate import evaluate_model
from src.save_model import save_model
from src.logger import logger

def train_model():
    df = pd.read_csv("../data/backorder.csv")
    logger.info("Data loaded successfully")
    df = preprocess_data(df)
    logger.info("Data preprocessed successfully")
    X_train, X_test, y_train, y_test = split_data(df)
    logger.info("Data split successfully")
    model = train_decision_tree(X_train, y_train)
    logger.info("Model trained successfully")
    evaluate_model(model, X_test, y_test)
    logger.info("Model evaluated successfully")
    save_model(model, "../models/backorder_model_v1.pkl")
    logger.info("Model saved successfully")
    logger.info("Training completed successfully")

if __name__ == "__main__":
    train_model()