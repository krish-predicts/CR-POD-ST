"""Portfolio-level stress testing and ECL aggregation."""

from __future__ import annotations

import numpy as np

from backend.data.credit_loader import sample_portfolio
from backend.data.macro_loader import load_macro_scenarios
from backend.stress.scenarios import apply_pd_shock
from ML.predict import predict_pd
from ML.risk_scoring import compute_ecl, ifrs9_stage
from ML.schemas.models import (
    LoanInput,
    PortfolioStressResult,
    ScenarioComparison,
    StressTestRequest,
    StressTestResponse,
)


def _loans_to_inputs(loans_df) -> list[LoanInput]:
    drop_cols = [c for c in ["loan_status"] if c in loans_df.columns]
    records = loans_df.drop(columns=drop_cols).to_dict(orient="records")
    return [LoanInput(**r) for r in records]


def run_portfolio_stress_test(request: StressTestRequest) -> StressTestResponse:
    scenarios_map = load_macro_scenarios()
    selected = [s for s in request.scenarios if s in scenarios_map]
    if not selected:
        raise ValueError(f"No valid scenarios found. Available: {list(scenarios_map)}")

    if request.loans:
        loans_df = __import__("pandas").DataFrame([l.model_dump() for l in request.loans])
    else:
        loans_df = sample_portfolio(n=request.sample_size or 500)

    loan_inputs = _loans_to_inputs(loans_df)
    predictions = predict_pd(loan_inputs, model_name=request.model_name, lgd=request.lgd)
    base_pds = np.array([p.pd for p in predictions])
    eads = loans_df["loan_amnt"].values.astype(float)

    results: list[PortfolioStressResult] = []
    normal_ecl: float | None = None

    for scenario_name in selected:
        scenario = scenarios_map[scenario_name]
        stressed_pds = apply_pd_shock(base_pds, scenario)

        stages = [ifrs9_stage(float(pd)) for pd in stressed_pds]
        ecls = [compute_ecl(float(pd), float(ead), stage, request.lgd) for pd, ead, stage in zip(stressed_pds, eads, stages, strict=False)]
        els = stressed_pds * request.lgd * eads

        stage_counts = {1: 0, 2: 0, 3: 0}
        stage_ecl = {1: 0.0, 2: 0.0, 3: 0.0}
        for stage, ecl in zip(stages, ecls, strict=False):
            stage_counts[stage] += 1
            stage_ecl[stage] += ecl

        total_ecl = float(sum(ecls))
        if scenario_name == "Normal":
            normal_ecl = total_ecl

        results.append(
            PortfolioStressResult(
                scenario=scenario_name,
                avg_pd=float(np.mean(stressed_pds)),
                total_el=float(np.sum(els)),
                total_ecl=total_ecl,
                stage_1_count=stage_counts[1],
                stage_2_count=stage_counts[2],
                stage_3_count=stage_counts[3],
                stage_1_ecl=stage_ecl[1],
                stage_2_ecl=stage_ecl[2],
                stage_3_ecl=stage_ecl[3],
                pd_multiplier=scenario.portfolio_pd_multiplier,
            )
        )

    comparison = [
        ScenarioComparison(
            scenario=r.scenario,
            avg_pd=r.avg_pd,
            total_ecl=r.total_ecl,
            delta_ecl_vs_normal=(r.total_ecl - normal_ecl) if normal_ecl is not None and r.scenario != "Normal" else 0.0 if r.scenario == "Normal" else None,
        )
        for r in results
    ]

    return StressTestResponse(results=results, comparison=comparison, loan_count=len(loan_inputs))
