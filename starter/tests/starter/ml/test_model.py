import numpy as np

from starter.ml.model import train_model, compute_model_metrics, predict, save_model, load_model
import pathlib

import pandas as pd
import pytest
from lightautoml.automl.base import AutoML

from starter.ml.data import load_data


@pytest.fixture(scope="module")
def model(data_file: pathlib.Path):
    data = load_data(data_file)
    auto_ml = train_model(data, timeout=10)  # quick training
    return auto_ml


@pytest.fixture(scope="module")
def data_point_df():
    data = pd.DataFrame(
        {
            "age": [39],
            "workclass": ["State-gov"],
            "fnlgt": [20885],
            "education": ["Bachelors"],
            "education-num": [13],
            "marital-status": ["Never-married"],
            "occupation": ["Adm-clerical"],
            "relationship": ["Not-in-family"],
            "race": ["White"],
            "sex": ["Male"],
            "capital-gain": [2174],
            "capital-loss": [0],
            "hours-per-week": [40],
            "native-country": ["United-States"],
        }
    )
    return data


def test_train_model(model):
    assert isinstance(model, AutoML)

    assert model.task.name == "binary"
    assert model.reader.target == "salary"
    assert model.reader.roles.keys() == {
        "age",
        "workclass",
        "fnlgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
    }


def test_compute_model_metrics():
    p, r, f1 = compute_model_metrics(np.array([1.0]), np.array([1.0]))

    assert p == 1.0
    assert r == 1.0
    assert f1 == 1.0


def test_predict(model, data_point_df):
    prediction = predict(model, data_point_df)
    assert prediction.shape == (len(data_point_df),)


def test_save_load_model(model, tmp_path: pathlib.Path):
    model_path = tmp_path / "model.pkl"
    save_model(model, str(model_path))
    assert model_path.exists()

    auto_ml_loaded = load_model(str(model_path))
    assert isinstance(auto_ml_loaded, AutoML)
