"""Training pipeline for Logistic Regression and XGBoost PD models."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from ML.features.engineering import prepare_training_frame
from ML.features.preprocessing import fit_preprocessor
from ML.schemas.models import ModelMetrics, TrainConfig, TrainResponse
from ML.utils import compute_metrics, get_env_path, save_artifacts, save_json, set_seed


def _feature_importance_lr(model, feature_names: list[str]) -> dict[str, float]:
    coefs = np.abs(model.coef_[0])
    total = coefs.sum() or 1.0
    return {name: float(val / total) for name, val in zip(feature_names, coefs, strict=False)}


def _feature_importance_xgb(model, feature_names: list[str]) -> dict[str, float]:
    importances = model.feature_importances_
    total = importances.sum() or 1.0
    return {name: float(val / total) for name, val in zip(feature_names, importances, strict=False)}


def _get_transformed_feature_names(preprocessor, raw_features: list[str]) -> list[str]:
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        return raw_features


def train_models(config: TrainConfig | None = None, csv_path: str | None = None) -> TrainResponse:
    config = config or TrainConfig()
    set_seed(config.random_state)

    data_path = csv_path or str(get_env_path("DATA_CSV_PATH", "data/credit_risk_dataset_new.csv"))
    model_dir = get_env_path("MODEL_DIR", "backend/models")

    df = pd.read_csv(data_path)
    X, y, feature_cols = prepare_training_frame(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.test_size, random_state=config.random_state, stratify=y
    )

    preprocessor = fit_preprocessor(X_train)
    X_train_t = preprocessor.transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    pos_weight = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)

    lr_model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=config.random_state)
    lr_model.fit(X_train_t, y_train)
    lr_prob = lr_model.predict_proba(X_test_t)[:, 1]
    lr_metrics = compute_metrics(y_test.values, lr_prob)

    xgb_model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=pos_weight,
        eval_metric="logloss",
        random_state=config.random_state,
        n_jobs=-1,
    )
    xgb_model.fit(X_train_t, y_train)
    xgb_prob = xgb_model.predict_proba(X_test_t)[:, 1]
    xgb_metrics = compute_metrics(y_test.values, xgb_prob)

    transformed_names = _get_transformed_feature_names(preprocessor, feature_cols)
    lr_importance = _feature_importance_lr(lr_model, transformed_names[: len(lr_model.coef_[0])])
    xgb_importance = _feature_importance_xgb(xgb_model, transformed_names[: len(xgb_model.feature_importances_)])

    best_model = "xgb" if xgb_metrics["roc_auc"] >= lr_metrics["roc_auc"] else "lr"
    best_importance = xgb_importance if best_model == "xgb" else lr_importance

    metadata: dict[str, Any] = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "best_model": best_model,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "feature_columns": feature_cols,
        "csv_path": data_path,
    }

    save_artifacts(
        model_dir,
        {
            "preprocessor.joblib": preprocessor,
            "lr_model.joblib": lr_model,
            "xgb_model.joblib": xgb_model,
            "training_metadata.json": metadata,
            "metrics.json": {"lr": lr_metrics, "xgb": xgb_metrics},
            "feature_importance.json": {"lr": lr_importance, "xgb": xgb_importance},
        },
    )
    save_json(model_dir / "training_metadata.json", metadata)

    return TrainResponse(
        best_model=best_model,
        metrics={
            "lr": ModelMetrics(model_name="lr", **lr_metrics),
            "xgb": ModelMetrics(model_name="xgb", **xgb_metrics),
        },
        feature_importance=best_importance,
        training_metadata=metadata,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train credit risk PD models")
    parser.add_argument("--csv", default=None, help="Path to training CSV")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    result = train_models(
        TrainConfig(test_size=args.test_size, random_state=args.seed),
        csv_path=args.csv,
    )
    print(f"Best model: {result.best_model}")
    for name, metrics in result.metrics.items():
        print(f"{name}: ROC AUC={metrics.roc_auc:.4f}, KS={metrics.ks:.4f}, Gini={metrics.gini:.4f}")


if __name__ == "__main__":
    main()
