"""Utilities for metrics, paths, and artifact I/O."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_project_root() -> Path:
    return PROJECT_ROOT


def get_env_path(key: str, default: str) -> Path:
    value = os.getenv(key, default)
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)


def ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))


def gini_coefficient(roc_auc: float) -> float:
    return 2 * roc_auc - 1


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    roc_auc = roc_auc_score(y_true, y_prob)
    return {
        "roc_auc": float(roc_auc),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "ks": ks_statistic(y_true, y_prob),
        "gini": gini_coefficient(roc_auc),
    }


def confusion_matrix_dict(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, int]:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def save_artifacts(model_dir: Path, artifacts: dict[str, Any]) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    for name, obj in artifacts.items():
        if name.endswith(".json"):
            save_json(model_dir / name, obj)
        else:
            joblib.dump(obj, model_dir / name)


def load_artifacts(model_dir: Path) -> dict[str, Any]:
    return {
        "preprocessor": joblib.load(model_dir / "preprocessor.joblib"),
        "lr_model": joblib.load(model_dir / "lr_model.joblib"),
        "xgb_model": joblib.load(model_dir / "xgb_model.joblib"),
        "metadata": load_json(model_dir / "training_metadata.json"),
    }


def models_exist(model_dir: Path) -> bool:
    required = ["preprocessor.joblib", "lr_model.joblib", "xgb_model.joblib", "training_metadata.json"]
    return all((model_dir / f).exists() for f in required)
