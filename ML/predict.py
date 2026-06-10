"""Inference for credit risk PD models."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from ML.features.engineering import NUMERIC_FEATURES, CATEGORICAL_FEATURES, engineer_features, loan_input_to_frame
from ML.risk_scoring import enrich_prediction
from ML.schemas.models import LoanInput, PredictionOutput
from ML.utils import get_env_path, load_artifacts, models_exist


def _get_model(artifacts: dict, model_name: str):
    if model_name == "lr":
        return artifacts["lr_model"]
    return artifacts["xgb_model"]


def _prepare_features(loans: list[dict] | pd.DataFrame, preprocessor) -> np.ndarray:
    if isinstance(loans, pd.DataFrame):
        frame = loans
    else:
        frame = pd.DataFrame(loans)
    engineered = engineer_features(frame)
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    return preprocessor.transform(engineered[feature_cols])


def predict_pd(
    loan: LoanInput | list[LoanInput] | pd.DataFrame,
    model_name: Literal["xgb", "lr"] = "xgb",
    lgd: float = 0.45,
    model_dir: str | None = None,
) -> PredictionOutput | list[PredictionOutput]:
    from pathlib import Path

    if model_dir is None:
        path = get_env_path("MODEL_DIR", "backend/models")
    else:
        path = Path(model_dir)
        if not path.is_absolute():
            path = get_env_path("MODEL_DIR", str(model_dir))
    if not models_exist(path):
        raise FileNotFoundError(f"No trained models found in {path}. Run training first.")

    artifacts = load_artifacts(path)
    model = _get_model(artifacts, model_name)
    preprocessor = artifacts["preprocessor"]

    if isinstance(loan, pd.DataFrame):
        loans_df = loan
        single = False
    elif isinstance(loan, list):
        loans_df = pd.DataFrame([l.model_dump() for l in loan])
        single = False
    else:
        loans_df = loan_input_to_frame(loan.model_dump())
        single = True

    X = _prepare_features(loans_df, preprocessor)
    probs = model.predict_proba(X)[:, 1]

    outputs: list[PredictionOutput] = []
    for idx, pd_value in enumerate(probs):
        enriched = enrich_prediction(float(pd_value), float(loans_df.iloc[idx]["loan_amnt"]), lgd=lgd)
        outputs.append(PredictionOutput(model_name=model_name, **enriched))

    return outputs[0] if single else outputs
