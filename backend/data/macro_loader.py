"""Load macro economic stress test scenarios from Excel."""

from __future__ import annotations

import pandas as pd

from ML.schemas.models import MacroVariable, StressScenario
from ML.utils import get_env_path

REQUIRED_COLUMNS = {"variable", "base_value", "stressed_value", "pd_multiplier"}


def _parse_sheet(name: str, df: pd.DataFrame) -> StressScenario:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Sheet '{name}' missing columns: {missing}")

    variables = [
        MacroVariable(
            variable=row["variable"],
            base_value=float(row["base_value"]),
            stressed_value=float(row["stressed_value"]),
            pd_multiplier=float(row["pd_multiplier"]),
        )
        for _, row in df.iterrows()
    ]
    portfolio_multiplier = float(df["pd_multiplier"].mean())
    return StressScenario(name=name, variables=variables, portfolio_pd_multiplier=portfolio_multiplier)


def load_macro_scenarios(xlsx_path: str | None = None) -> dict[str, StressScenario]:
    path = xlsx_path or str(get_env_path("MACRO_XLSX_PATH", "data/US_Macro_Economic_Stress_Test_Data.xlsx"))
    sheets = pd.read_excel(path, sheet_name=None)
    return {name: _parse_sheet(name, sheet) for name, sheet in sheets.items()}
