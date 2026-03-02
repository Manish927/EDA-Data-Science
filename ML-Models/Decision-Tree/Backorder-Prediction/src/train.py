import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import argparse
import pandas as pd

from preprocessing import preprocess_data
from preprocessing import split_data

from train_decision_tree import train_decision_tree
from advanced_models.xgboost_model import train_xgboost

from evaluate import evaluate_model
from save_model import save_model
from logger import logger


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        default="decision_tree",
        choices=["decision_tree", "xgboost"],
        help="Model to train"
    )

    URL = 'https://raw.githubusercontent.com/Manish927/EDA-Data-Science/refs/heads/main/ML-Models/Decision-Tree/Backorder-Prediction/data/backorder.csv'
    args = parser.parse_args()
    logger.info("Loading dataset")
    df = pd.read_csv(URL)
    logger.info("Preprocessing data")
    df = preprocess_data(df)
    logger.info("Splitting data")
    X_train, X_test, y_train, y_test = split_data(df)

    if args.model == "decision_tree":
        logger.info("Training Decision Tree model")
        model = train_decision_tree(X_train, y_train)
        model_name = "decision_tree_model.pkl"
    elif args.model == "xgboost":
        logger.info("Training XGBoost model")
        model = train_xgboost(X_train, y_train)
        model_name = "xgboost_model.pkl"

    logger.info("Evaluating model")
    evaluate_model(model, X_test, y_test)
    logger.info("Saving model")
    save_model(model, f"models/{model_name}")
    logger.info("Training pipeline completed successfully")


if __name__ == "__main__":
    main()