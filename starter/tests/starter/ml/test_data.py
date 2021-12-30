import numpy as np

from starter.ml.data import process_data


def test_process_data(train_data):
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
        train_data, categorical_features=cat_features, label="salary", training=True
    )

    assert isinstance(X_train, np.ndarray)
    assert X_train.shape[0] == train_data.shape[0]
    assert X_train.shape[1] == 108
    assert isinstance(y_train, np.ndarray)
