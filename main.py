import enum
import os
import pathlib
import subprocess

import pandas as pd
import pydantic
from fastapi import FastAPI
from loguru import logger

from starter.ml.model import load_model, predict


app = FastAPI()


@app.on_event("startup")
def init():
    """Startup event that pulls from dvc if necessary and loads model into app state."""
    if "DYNO" in os.environ and pathlib.Path(".dvc").exists():
        logger.info("Running on Heroku. Performing DVC pull...")
        subprocess.call(["dvc", "config", "core.no_scm", "true"])
        return_value = subprocess.call(["dvc", "pull"])
        if return_value != 0:
            logger.error("DVC pull failed. Exiting...")
            exit(return_value)
        logger.info("DVC pull successful.")
    app.state.model = load_model()


class Values(enum.Enum):
    over: str = ">50K"
    under: str = "<=50K"

    @classmethod
    def from_int(cls, value: int) -> str:
        return Values.under if value == 0 else Values.over


class PredictionRequestResponse(pydantic.BaseModel):
    prediction: Values


class PredictionRequestBody(pydantic.BaseModel):
    age: int = pydantic.Field(..., example=39)
    workclass: str = pydantic.Field(..., example="State-gov")
    fnlgt: int = pydantic.Field(..., example=20885)
    education: str = pydantic.Field(..., example="Bachelors")
    education_num: int = pydantic.Field(..., example=13, alias="education-num")
    marital_status: str = pydantic.Field(..., example="Never-married", alias="marital-status")
    occupation: str = pydantic.Field(..., example="Adm-clerical")
    relationship: str = pydantic.Field(..., example="Not-in-family")
    race: str = pydantic.Field(..., example="White")
    sex: str = pydantic.Field(..., example="Male")
    capital_gain: int = pydantic.Field(..., example=2174, alias="capital-gain")
    capital_loss: int = pydantic.Field(..., example=0, alias="capital-loss")
    hours_per_week: int = pydantic.Field(..., example=40, alias="hours-per-week")
    native_country: str = pydantic.Field(..., example="United-States", alias="native-country")


@app.get("/")
async def get_root():
    """Greetings endpoint."""
    return {"message": "Greetings!"}


@app.post("/predict", response_model=PredictionRequestResponse)
async def post_predict(body: PredictionRequestBody) -> PredictionRequestResponse:
    """Perform prediction on given input."""
    return PredictionRequestResponse(
        prediction=Values.from_int(int(predict(app.state.model, pd.DataFrame(body.dict(), index=[0]))))
    )
