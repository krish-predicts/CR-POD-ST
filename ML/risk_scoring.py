"""Risk scorecards, EL, and IFRS 9 staging."""

from __future__ import annotations

import os

import numpy as np

DEFAULT_LGD = float(os.getenv("DEFAULT_LGD", "0.45"))


def pd_to_score(pd_value: float, min_score: int = 300, max_score: int = 850) -> int:
    """Map PD to an inverse credit score (lower PD = higher score)."""
    clipped = float(np.clip(pd_value, 1e-6, 0.9999))
    score = max_score - (max_score - min_score) * clipped
    return int(round(score))


def pd_to_risk_band(pd_value: float) -> str:
    if pd_value < 0.10:
        return "Low"
    if pd_value < 0.20:
        return "Medium"
    if pd_value < 0.35:
        return "High"
    return "Very High"


def ifrs9_stage(pd_value: float, significant_deterioration: bool = False) -> int:
    if pd_value >= 0.20 or significant_deterioration and pd_value >= 0.05:
        if pd_value >= 0.20:
            return 3
        return 2
    if pd_value >= 0.05:
        return 2
    return 1


def compute_expected_loss(pd_value: float, ead: float, lgd: float = DEFAULT_LGD) -> float:
    return float(pd_value * lgd * ead)


def compute_ecl(pd_value: float, ead: float, stage: int, lgd: float = DEFAULT_LGD) -> float:
    """Simplified IFRS 9 ECL: Stage 1 uses 12-month PD proxy; Stage 2/3 lifetime."""
    if stage == 1:
        horizon_pd = min(pd_value, 0.99)
    else:
        horizon_pd = min(pd_value * 1.5, 0.99)
    return compute_expected_loss(horizon_pd, ead, lgd)


def enrich_prediction(pd_value: float, loan_amnt: float, lgd: float = DEFAULT_LGD) -> dict:
    stage = ifrs9_stage(pd_value)
    el = compute_expected_loss(pd_value, loan_amnt, lgd)
    ecl = compute_ecl(pd_value, loan_amnt, stage, lgd)
    return {
        "pd": float(pd_value),
        "risk_score": pd_to_score(pd_value),
        "risk_band": pd_to_risk_band(pd_value),
        "predicted_class": int(pd_value >= 0.5),
        "expected_loss": el,
        "ifrs9_stage": stage,
        "ecl": ecl,
    }
