"""Load and clean credit risk CSV data."""

from __future__ import annotations

import pandas as pd

from ML.features.engineering import clean_raw_data
from ML.utils import get_env_path


REQUIRED_COLUMNS = [
    "person_age",
    "person_income",
    "person_home_ownership",
    "person_emp_length",
    "loan_intent",
    "loan_grade",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_default_on_file",
    "cb_person_cred_hist_length",
]


def load_credit_data(csv_path: str | None = None) -> pd.DataFrame:
    path = csv_path or str(get_env_path("DATA_CSV_PATH", "data/credit_risk_dataset_new.csv"))
    df = pd.read_csv(path)
    df = clean_raw_data(df)

    for col in REQUIRED_COLUMNS:
        if col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].fillna("unknown")
            else:
                df[col] = df[col].fillna(df[col].median())

    return df.dropna(subset=REQUIRED_COLUMNS).reset_index(drop=True)


def sample_portfolio(n: int = 500, csv_path: str | None = None, random_state: int = 42) -> pd.DataFrame:
    df = load_credit_data(csv_path)
    n = min(n, len(df))
    return df.sample(n=n, random_state=random_state).reset_index(drop=True)
