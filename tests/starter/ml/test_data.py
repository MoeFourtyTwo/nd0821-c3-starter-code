import pathlib

import pandas as pd

from starter.ml.data import load_data, convert_salary_column, split_data, find_slices


def test_load_data(data_file: pathlib.Path) -> None:
    data = load_data(data_file)

    assert len(data) > 10000


def test_convert_salary_column(data_file: pathlib.Path) -> None:
    data = pd.read_csv(data_file)

    assert data["salary"].dtype == "object"
    data = convert_salary_column(data)
    assert data["salary"].dtype == "float64"


def test_split_data(data_file: pathlib.Path) -> None:
    data = pd.read_csv(data_file)

    train, test = split_data(data)

    assert len(train) + len(test) == len(data)
    assert len(train) > 0.8 / 0.2 * len(test) - 5  # offset of 5 for rounding errors


def test_find_slcices(data_file: pathlib.Path) -> None:
    data = pd.read_csv(data_file)

    slices = find_slices(data)

    assert len(slices) == 8
