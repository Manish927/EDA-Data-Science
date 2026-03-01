
import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(path):
    df = pd.read_csv(path)
    return df


def preprocess_data(df):

    # Handle missing values
    df['lead_time'] = df['lead_time'].fillna(df['lead_time'].median())

    # Identify categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns

    # One hot encoding
    df = pd.get_dummies(
        df,
        columns=categorical_cols,
        drop_first=True
    )

    return df


def split_data(df):

    X = df.drop("went_on_backorder_Yes", axis=1)
    y = df["went_on_backorder_Yes"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    return X_train, X_test, y_train, y_test
