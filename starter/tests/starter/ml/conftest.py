import pathlib

import pandas as pd
import pytest


@pytest.fixture(scope="session")
def data_dir():
    return pathlib.Path(__file__).parent.parent.parent.parent / "data"


@pytest.fixture(scope="function")
def train_data(data_dir):
    return pd.read_csv(data_dir / "census_cleaned.csv")
