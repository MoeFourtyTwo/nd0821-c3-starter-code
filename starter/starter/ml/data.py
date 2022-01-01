from __future__ import annotations

import pathlib
from typing import Optional

import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(path: str | pathlib.Path = "../data/census_cleaned.csv") -> pd.DataFrame:
    """Loads the data from the given path.
    Args:
        path: Path to the data.

    Returns:
        DataFrame containing the data. With salary column converted to numeric.
    """
    data = pd.read_csv(path)
    data = convert_salary_column(data)

    return data


def convert_salary_column(data: pd.DataFrame) -> pd.DataFrame:
    """Convert salary column to a numeric column.
    0.0 represents a salary of less than $50,000.
    1.0 represents a salary of $50,000 or more.
    Args:
        data: DataFrame containing the salary column.

    Returns:
        DataFrame with salary column converted to numeric.
    """
    data["salary"] = (data["salary"] != "<=50K").astype(float)
    return data


def split_data(data: pd.DataFrame, train_size: float = 0.8) -> (pd.DataFrame, pd.DataFrame):
    """Splits the data into training and test sets.
    Args:
        data: DataFrame containing the data.
        train_size: Percentage of data to be used for training.

    Returns:
        Tuple containing training and test data.
    """
    train_data, test_data = train_test_split(data, test_size=1.0 - train_size, random_state=42)
    return train_data, test_data


def get_slice(data: pd.DataFrame, column: str, value: str) -> pd.DataFrame:
    """Get a slice of the data based on the given column and value.
    Args:
        data: DataFrame containing the data.
        column: Column to slice on.
        value: Value to slice on.

    Returns:
        DataFrame containing only the rows that match the given column and value.
    """
    return data[data[column] == value]


def find_slices(data: pd.DataFrame, exclude: Optional[list] = None) -> dict[str, list[str]]:
    exclude = exclude or ["salary"]

    slices = {
        col: list(data[col].unique()) for col in data.columns if col not in exclude and data[col].dtype == "object"
    }

    return slices
