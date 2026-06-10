"""Scenario PD adjustment logic."""

from __future__ import annotations

import numpy as np

from ML.schemas.models import StressScenario


def apply_pd_shock(base_pd: float | np.ndarray, scenario: StressScenario) -> float | np.ndarray:
    shocked = np.asarray(base_pd) * scenario.portfolio_pd_multiplier
    return np.clip(shocked, 0, 0.9999)
