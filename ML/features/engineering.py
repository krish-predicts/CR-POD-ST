"""Feature engineering and data cleaning for credit risk PD models."""

from __future__ import annotations

import numpy as np
import pandas as pd

RAW_COLUMNS = [
    "person_age",
    "person_income",
    "person_home_ownership",
    "person_emp_length",
    "loan_intent",
    "loan_grade",
    "loan_amnt",
    "loan_int_rate",
    "loan_status",
    "loan_percent_income",
    "cb_person_default_on_file",
    "cb_person_cred_hist_length",
]

CATEGORICAL_FEATURES = [
    "person_home_ownership",
    "loan_intent",
    "loan_grade",
    "cb_person_default_on_file",
    "age_bucket",
    "income_bucket",
]

NUMERIC_FEATURES = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
    "log_person_income",
    "dti_ratio",
    "loan_grade_ord",
]

GRADE_ORDINAL = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean outliers and invalid values in raw credit data."""
    data = df.copy()

    if "person_age" in data.columns:
        data["person_age"] = data["person_age"].clip(lower=18, upper=100)

    if "person_emp_length" in data.columns:
        data["person_emp_length"] = pd.to_numeric(data["person_emp_length"], errors="coerce")
        data["person_emp_length"] = data["person_emp_length"].clip(lower=0, upper=50)

    if "person_income" in data.columns:
        data["person_income"] = pd.to_numeric(data["person_income"], errors="coerce")
        data.loc[data["person_income"] <= 0, "person_income"] = np.nan

    if "loan_amnt" in data.columns:
        data["loan_amnt"] = pd.to_numeric(data["loan_amnt"], errors="coerce")
        data.loc[data["loan_amnt"] <= 0, "loan_amnt"] = np.nan

    return data


def _age_bucket(age: float) -> str:
    if age < 25:
        return "young"
    if age < 35:
        return "mid_young"
    if age < 50:
        return "mid"
    if age < 65:
        return "senior"
    return "elder"


def _income_bucket(income: float) -> str:
    if income < 30000:
        return "low"
    if income < 60000:
        return "lower_mid"
    if income < 100000:
        return "upper_mid"
    return "high"


def engineer_features(df: pd.DataFrame, income_quantiles: dict[str, float] | None = None) -> pd.DataFrame:
    """Apply feature engineering on cleaned data."""
    data = clean_raw_data(df)

    data["log_person_income"] = np.log1p(data["person_income"].fillna(data["person_income"].median()))
    data["dti_ratio"] = data["loan_percent_income"].fillna(
        data["loan_amnt"] / data["person_income"].replace(0, np.nan)
    )
    data["dti_ratio"] = data["dti_ratio"].clip(lower=0, upper=1)

    data["age_bucket"] = data["person_age"].apply(_age_bucket)
    if income_quantiles:
        data["income_bucket"] = pd.cut(
            data["person_income"],
            bins=[-np.inf, income_quantiles["q25"], income_quantiles["q50"], income_quantiles["q75"], np.inf],
            labels=["low", "lower_mid", "upper_mid", "high"],
        ).astype(str)
    else:
        data["income_bucket"] = data["person_income"].apply(_income_bucket)

    data["loan_grade_ord"] = data["loan_grade"].map(GRADE_ORDINAL).fillna(4)

    for col in CATEGORICAL_FEATURES:
        if col in data.columns:
            data[col] = data[col].astype(str).replace("nan", "unknown")

    return data


def prepare_training_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Return feature matrix, target, and feature name list."""
    engineered = engineer_features(df)
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = engineered[feature_cols].copy()
    y = engineered["loan_status"].astype(int)
    return X, y, feature_cols


def loan_input_to_frame(loan: dict) -> pd.DataFrame:
    """Convert a single loan dict to a one-row DataFrame."""
    return pd.DataFrame([loan])
