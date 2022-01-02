import pathlib

import pytest


@pytest.fixture(scope="session")
def data_file():
    return pathlib.Path(__file__).parent.parent.parent.parent / "data" / "census_cleaned.csv"
