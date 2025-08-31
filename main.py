import os
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from ml.data import apply_label, process_data
from ml.model import load_model, inference

# DO NOT MODIFY


class Data(BaseModel):
    age: int = Field(..., example=37)
    workclass: str = Field(..., example="Private")
    fnlgt: int = Field(..., example=178356)
    education: str = Field(..., example="HS-grad")
    education_num: int = Field(..., example=10, alias="education-num")
    marital_status: str = Field(
        ..., example="Married-civ-spouse", alias="marital-status"
    )
    occupation: str = Field(..., example="Prof-specialty")
    relationship: str = Field(..., example="Husband")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., example=0, alias="capital-gain")
    capital_loss: int = Field(..., example=0, alias="capital-loss")
    hours_per_week: int = Field(..., example=40, alias="hours-per-week")
    native_country: str = Field(..., example="United-States", alias="native-country")


# ----- Artifacts -----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENC_PATH = os.path.join(BASE_DIR, "model", "encoder.pkl")
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")

try:
    encoder = load_model(ENC_PATH)
    model = load_model(MODEL_PATH)
except Exception as e:
    # Fail fast with a clear message if artifacts are missing
    raise RuntimeError(f"Failed to load artifacts: {e}")

# ----- App -----
app = FastAPI(title="Income Classifier API")

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

@app.get("/")
async def get_root():
    return {"message": "hello!"}  # keep this to match your current output

@app.post("/data/")
async def post_inference(data: Data):
    try:
        # Prepare a one-row DataFrame with hyphenated column names
        data_dict = data.dict()
        row = {k.replace("_", "-"): [v] for k, v in data_dict.items()}
        df = pd.DataFrame.from_dict(row)

        X, _, _, _ = process_data(
            df,
            categorical_features=cat_features,
            label=None,
            training=False,
            encoder=encoder,
            lb=None,
        )
        pred = inference(model, X)                # np.array([0 or 1])
        return {"result": apply_label(pred)}      # ">50K" or "<=50K"
    except Exception as e:
        # Return JSON error so the client can print it (no HTML 500 page)
        raise HTTPException(status_code=400, detail=f"Inference error: {e}")