import fire as fire
import pandas as pd
from lightautoml.automl.base import AutoML
from loguru import logger

from starter.ml.data import load_data, split_data, find_slices, get_slice
from starter.ml.model import predict, compute_model_metrics, train_model, save_model, load_model


def main(load_saved_model: bool = False) -> None:
    data = load_data()
    train_df, test_df = split_data(data)

    if load_saved_model:
        logger.info("Loading saved model...")
        model = load_model("../model/model.pkl")
    else:
        logger.info("Training model from scratch...")
        model = train_model(train_df)
        save_model(model, "../model/model.pkl")

    logger.info("Generating report...")
    generate_report(model, test_df)


def generate_report(model: AutoML, test_df: pd.DataFrame) -> None:
    predictions = predict(model, test_df)
    precision, recall, fbeta = compute_model_metrics(test_df["salary"], predictions)

    report = pd.DataFrame(columns=["Category", "Value", "Samples", "Precision", "Recall", "F-beta"])
    report = report.append(
        {"Samples": len(test_df), "Precision": precision, "Recall": recall, "F-beta": fbeta}, ignore_index=True
    )
    for category, values in find_slices(test_df).items():
        for value in values:
            slice_df = get_slice(test_df, category, value)
            predictions = predict(model, slice_df)
            precision, recall, fbeta = compute_model_metrics(slice_df["salary"], predictions)
            report = report.append(
                {
                    "Category": category,
                    "Value": value,
                    "Samples": len(slice_df),
                    "Precision": precision,
                    "Recall": recall,
                    "F-beta": fbeta,
                },
                ignore_index=True,
            )
    with open("../model/slice_output.txt", "w") as f:
        f.write(report.to_markdown())
        logger.info("Report generated at ../model/slice_output.txt")


if __name__ == "__main__":
    fire.Fire(main)
