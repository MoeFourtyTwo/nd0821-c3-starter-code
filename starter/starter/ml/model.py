import joblib
import numpy as np
import pandas as pd
from lightautoml.automl.base import AutoML
from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.dataset.roles import NumericRole, CategoryRole
from lightautoml.tasks import Task
from lightautoml.tasks.common_metric import BestClassBinaryWrapper, F1Factory
from sklearn.metrics import fbeta_score, precision_score, recall_score


def train_model(train_df: pd.DataFrame, target: str = "salary", timeout: int = 3600) -> AutoML:
    """Trains a AutoML model based on the given train dataframe.

    Args:
        train_df: Dataframe containing the training data.
        target: The target column to train on. Defaults to "salary".
        timeout: Timeout in seconds. Defaults to 3600.

    Returns:
        Trained AutoML model.
    """
    automl = TabularAutoML(task=Task(name="binary", metric=BestClassBinaryWrapper(F1Factory())), timeout=timeout)
    automl.fit_predict(
        train_df,
        roles={
            "target": target,
            NumericRole(): ["age", "capital-gain", "capital-loss"],
            CategoryRole(encoding_type="ohe"): [
                "workclass",
                "education",
                "marital-status",
                "occupation",
                "relationship",
                "race",
                "sex",
                "native-country",
            ],
        },
        verbose=2,
    )

    return automl


def compute_model_metrics(y_true, y_pred) -> tuple[float, float, float]:
    """Validates the trained machine learning model using precision, recall, and F beta.

    Args:
        y_true: Known labels, binarized.
        y_pred: Predicted labels, binarized.
    Returns:
        Precision, recall, and F beta score.
    """
    precision = precision_score(y_true, y_pred, zero_division=1)
    recall = recall_score(y_true, y_pred, zero_division=1)
    fbeta = fbeta_score(y_true, y_pred, beta=1, zero_division=1)
    return precision, recall, fbeta


def predict(model: AutoML, data: pd.DataFrame) -> np.ndarray:
    """Run model inferences and return the predictions.
    Args:
        model: Trained AutoML model.
        data: Dataframe containing the data.

    Returns:
        Predictions.
    """
    predictions = model.predict(data)
    predictions = (np.squeeze(predictions.data, axis=-1) > 0.5) * 1.0
    return predictions


def save_model(model: AutoML, target: str) -> None:
    """Save the model to a file.
    Args:
        model: Trained machine learning model.
        target: Path to save the model to.

    Returns:
        None
    """
    joblib.dump(model, target)


def load_model(target: str) -> AutoML:
    """Loads a model from a file.
    Args:
        target: Path to save the model.

    Returns:
        Trained machine learning model.
    """
    return joblib.load(target)
