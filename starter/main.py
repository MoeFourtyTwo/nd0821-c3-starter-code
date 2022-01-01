import enum

import pandas as pd
import pydantic
from fastapi import FastAPI

from starter.ml.model import load_model, predict

app = FastAPI()


@app.on_event("startup")
def init_model():
    app.state.model = load_model("./model/model.pkl")


class Values(enum.Enum):
    over: str = ">50K"
    under: str = "<=50K"

    @classmethod
    def from_int(cls, value: int) -> str:
        if value == 0:
            return Values.under
        return Values.over


class InferenceRequestResponse(pydantic.BaseModel):
    prediction: Values


class InferenceRequestBody(pydantic.BaseModel):
    age: int = pydantic.Field(..., example=39)
    workclass: str = pydantic.Field(..., example="State-gov")
    fnlgt: int = pydantic.Field(..., example=20885, alias="fnlwgt")
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
async def root():
    return {"message": "Hello World"}


@app.post("/inference", response_model=InferenceRequestResponse)
async def inference(body: InferenceRequestBody) -> InferenceRequestResponse:
    return InferenceRequestResponse(
        prediction=Values.from_int(int(predict(app.state.model, pd.DataFrame(body.dict(), index=[0]))))
    )
