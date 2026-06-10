"""FastAPI backend for Credit Risk PD & Stress Testing Engine."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.stress.portfolio import run_portfolio_stress_test
from ML.predict import predict_pd
from ML.schemas.models import (
    LoanInput,
    PredictionOutput,
    StressTestRequest,
    StressTestResponse,
    TrainConfig,
    TrainResponse,
)
from ML.train import train_models
from ML.utils import get_env_path, load_json, models_exist

app = FastAPI(
    title="Credit Risk PD & Stress Testing Engine",
    description="PD modeling, stress testing, and regulatory ECL calculations",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_DIR = get_env_path("MODEL_DIR", "backend/models")


class PredictRequest(BaseModel):
    loan: LoanInput | None = None
    loans: list[LoanInput] | None = None
    model_name: Literal["xgb", "lr"] = "xgb"
    lgd: float = Field(default=0.45, ge=0, le=1)


class PredictResponse(BaseModel):
    predictions: list[PredictionOutput]


@app.get("/health")
def health() -> dict:
    loaded = models_exist(MODEL_DIR)
    metadata = {}
    if loaded:
        try:
            metadata = load_json(MODEL_DIR / "training_metadata.json")
        except Exception:
            metadata = {}
    return {
        "status": "ok",
        "model_loaded": loaded,
        "best_model": metadata.get("best_model"),
        "trained_at": metadata.get("trained_at"),
    }


@app.post("/train", response_model=TrainResponse)
def train(config: TrainConfig | None = None) -> TrainResponse:
    try:
        return train_models(config or TrainConfig())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/metrics")
def metrics() -> dict:
    metrics_path = MODEL_DIR / "metrics.json"
    importance_path = MODEL_DIR / "feature_importance.json"
    if not metrics_path.exists():
        raise HTTPException(status_code=404, detail="No metrics found. Train models first.")
    payload = {"metrics": load_json(metrics_path)}
    if importance_path.exists():
        payload["feature_importance"] = load_json(importance_path)
    return payload


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    if not models_exist(MODEL_DIR):
        raise HTTPException(status_code=400, detail="Models not trained. Call POST /train first.")

    try:
        if request.loans:
            preds = predict_pd(request.loans, model_name=request.model_name, lgd=request.lgd)
            if not isinstance(preds, list):
                preds = [preds]
        elif request.loan:
            pred = predict_pd(request.loan, model_name=request.model_name, lgd=request.lgd)
            preds = [pred] if isinstance(pred, PredictionOutput) else pred
        else:
            raise HTTPException(status_code=422, detail="Provide 'loan' or 'loans'")
        return PredictResponse(predictions=preds)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/stress_test", response_model=StressTestResponse)
def stress_test(request: StressTestRequest) -> StressTestResponse:
    if not models_exist(MODEL_DIR):
        raise HTTPException(status_code=400, detail="Models not trained. Call POST /train first.")
    try:
        return run_portfolio_stress_test(request)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
