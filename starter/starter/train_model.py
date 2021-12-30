# Script to train machine learning model.
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.

# Add code to load in the data.
from starter.ml.data import process_data
from starter.ml.model import train_model, inference, compute_model_metrics


def main():
    data = pd.read_csv("../data/census_cleaned.csv")
    # Optional enhancement, use K-fold cross validation instead of a train-test split.
    train, test = train_test_split(data, test_size=0.20)
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )
    # Proces the test data with the process_data function.
    X_test, y_test, *_ = process_data(
        test,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )
    # Train and save a model.
    tree = train_model(X_train, y_train)
    predict = inference(tree, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, predict)

    logger.info(f"Precision: {precision}")
    logger.info(f"Recall: {recall}")
    logger.info(f"F-beta: {fbeta}")


if __name__ == "__main__":
    main()
