import pandas as pd

from preprocessing import preprocess_data, split_data
from train_models import train_decision_tree
from evaluate import evaluate_model
from save_model import save_model
from logger import logger

def train_model():
    #df = pd.read_csv("../data/backorder.csv")
    URL = 'https://raw.githubusercontent.com/Manish927/EDA-Data-Science/refs/heads/main/ML-Models/Decision-Tree/Backorder-Prediction/data/backorder.csv'
    df = pd.read_csv(URL)
    logger.info("Data loaded successfully")
    df = preprocess_data(df)
    logger.info("Data preprocessed successfully")
    X_train, X_test, y_train, y_test = split_data(df)
    logger.info("Data split successfully")
    model = train_decision_tree(X_train, y_train)
    logger.info("Model trained successfully")
    evaluate_model(model, X_test, y_test)
    logger.info("Model evaluated successfully")
    save_model(model, "models/backorder_model_v1.pkl")
    logger.info("Model saved successfully")
    logger.info("Training completed successfully")

if __name__ == "__main__":
    train_model()