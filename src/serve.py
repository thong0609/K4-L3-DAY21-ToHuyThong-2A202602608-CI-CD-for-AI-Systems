"""Income prediction API; downloads the approved model from Amazon S3."""
from contextlib import asynccontextmanager
import os
from pathlib import Path

import boto3
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, FiniteFloat

FEATURE_NAMES = [
    "age", "workclass", "education_num", "marital_status", "occupation",
    "relationship", "sex", "capital_gain", "capital_loss", "hours_per_week",
]


def download_model():
    model_path = Path(os.getenv("MODEL_PATH", str(Path.home() / "income-api/models/model.joblib")))
    model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = model_path.with_suffix(".download")
    boto3.client("s3", region_name=os.getenv("AWS_DEFAULT_REGION", "ap-southeast-2")).download_file(
        os.environ["ARTIFACT_BUCKET"], "artifacts/current/model.joblib", str(temporary_path)
    )
    model = joblib.load(temporary_path)
    temporary_path.replace(model_path)
    return model


@asynccontextmanager
async def lifespan(app):
    app.state.model = download_model()
    yield


app = FastAPI(lifespan=lifespan)


class ScoreRequest(BaseModel):
    features: list[FiniteFloat]


@app.get("/healthz")
def healthz():
    if getattr(app.state, "model", None) is None:
        raise HTTPException(status_code=503, detail="Model is not ready")
    return {"status": "ok"}


@app.post("/score")
def score(req: ScoreRequest):
    if len(req.features) != len(FEATURE_NAMES):
        raise HTTPException(status_code=400, detail="Exactly 10 features are required")
    row = pd.DataFrame([req.features], columns=FEATURE_NAMES)
    prediction = int(app.state.model.predict(row)[0])
    return {"prediction": prediction, "label": "thu_nhap_cao" if prediction else "thu_nhap_thap"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
