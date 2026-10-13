from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from openpyxl import load_workbook

from q4_model import (
    N_DAYS,
    PolicyConfig,
    build_forecast_archive,
    build_inherited_risk_margins,
    load_data,
    run_policy,
    write_json,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXED_PRICE_PATH = PROJECT_ROOT / "data" / "附件1.xlsx"
OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "Q4"
    / "experiments"
    / "round2_inherited_price"
    / "metrics"
    / "inheritance_regression.json"
)


EXPECTED = {
    "q42": {
        "actual_total_cost_yuan": 13923903.386969224,
        "original_contract_kwh": 21605749.445191227,
        "emergency_purchase_kwh": 111928.37713393185,
        "initial_soc_evaluation_kwh": 2474.99712581418,
        "terminal_soc_evaluation_kwh": 1718.6737161078236,
    },
    "q43": {
        "actual_total_cost_yuan": 13833843.322638258,
        "original_contract_kwh": 21848677.8903264,
        "final_contract_kwh": 21057849.899175413,
        "adjustment_increase_kwh": 502453.2704979111,
        "adjustment_decrease_kwh": 1293281.261648901,
        "emergency_purchase_kwh": 51525.69845274206,
        "initial_soc_evaluation_kwh": 2492.6295222483786,
        "terminal_soc_evaluation_kwh": 1661.099380902042,
    },
}


def load_fixed_prices() -> np.ndarray:
    rows = list(load_workbook(FIXED_PRICE_PATH, read_only=True, data_only=True).active.values)
    price = np.asarray([row[1] for row in rows[1:]], dtype=float)
    if price.shape != (144,) or not np.isfinite(price).all() or np.any(price <= 0):
        raise ValueError("附件1固定电价向量无效")
    return price


def compare(actual: dict[str, float], expected: dict[str, float]) -> dict[str, float]:
    return {key: float(actual[key] - value) for key, value in expected.items()}


def main() -> None:
    data = load_data()
    fixed_data = replace(data, actual_price=np.tile(load_fixed_prices(), (N_DAYS, 1)))

    historical = build_forecast_archive(fixed_data, "historical")
    q42_config = PolicyConfig(
        "Q4_inheritance_regression_Q2_M3",
        "regression",
        "Q4-2",
        "historical",
        "inherited",
        False,
        alpha=0.8,
    )
    q42 = run_policy(
        fixed_data,
        historical,
        q42_config,
        margins=build_inherited_risk_margins(historical, "Q4-2", 0.8),
    )

    official = build_forecast_archive(fixed_data, "official")
    q43_config = PolicyConfig(
        "Q4_inheritance_regression_Q3_M1I",
        "regression",
        "Q4-3",
        "official",
        "inherited",
        True,
        alpha=0.8,
    )
    q43 = run_policy(
        fixed_data,
        official,
        q43_config,
        margins=build_inherited_risk_margins(official, "Q4-3", 0.8),
    )

    q42_residuals = compare(q42.metrics, EXPECTED["q42"])
    q43_residuals = compare(q43.metrics, EXPECTED["q43"])
    max_abs = max(abs(value) for value in [*q42_residuals.values(), *q43_residuals.values()])
    payload = {
        "schema_version": 1,
        "question_id": "Q4",
        "purpose": "prove that the corrected implementation reproduces the approved Q2/Q3 results when attachment-4 prices are replaced by the original fixed price vector",
        "fixed_price_source": FIXED_PRICE_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "q42_residuals": q42_residuals,
        "q43_residuals": q43_residuals,
        "maximum_absolute_residual": max_abs,
        "tolerance": 5e-5,
        "q42_hard_checks_pass": q42.validation["hard_checks_pass"],
        "q43_hard_checks_pass": q43.validation["hard_checks_pass"],
        "status": "PASS" if max_abs <= 5e-5 and q42.validation["hard_checks_pass"] and q43.validation["hard_checks_pass"] else "FAIL",
    }
    write_json(OUTPUT_PATH, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if payload["status"] != "PASS":
        raise RuntimeError(payload)


if __name__ == "__main__":
    raise SystemExit("请运行根目录 run.py --task full，以先生成本次主方案和基准结果。")
