from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from q2_common import (
    CHECK_TOLERANCE,
    DELTA_HOURS,
    ETA_CHARGE,
    ETA_DISCHARGE,
    EVALUATION_START_INDEX,
    FINAL_DAY_INDEX,
    INTERVAL_LIMIT_KWH,
    N_DAYS,
    N_INTERVALS,
    NONZERO_TOLERANCE,
    PRICE_PATH,
    ACTUAL_PATH,
    PROJECT_ROOT,
    RANDOM_SEED,
    SOC_INITIAL_KWH,
    SOC_MAX_KWH,
    SOC_MIN_KWH,
    DailyLP,
    DailySolve,
    ForecastBundle,
    Q2Data,
    StrategyConfig,
    StrategyRun,
    build_forecasts,
    build_risk_margins,
    emergency_events,
    environment_info,
    load_q2_data,
    sha256,
    specified_date_tables,
    storage_four_hour_summary,
)


ROUND_DIR = PROJECT_ROOT / "results" / "Q2" / "experiments" / "round3_full_feedback"
TABLE_DIR = ROUND_DIR / "tables"
METRIC_DIR = ROUND_DIR / "metrics"

MAIN_CONFIG = StrategyConfig(
    strategy_id="M2_daily_replan_actual_soc_feedback_alpha0.8_gamma0.5_exp",
    role="main",
    alpha=0.8,
    gamma=0.5,
    weight_mode="exponential",
    use_risk_margin=True,
)
BASELINE_CONFIG = StrategyConfig(
    strategy_id="B2_daily_replan_actual_soc_feedback_b0_gamma0.5_exp",
    role="usable_baseline",
    alpha=None,
    gamma=0.5,
    weight_mode="exponential",
    use_risk_margin=False,
)


def save_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.10f")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def cold_start_plan() -> DailySolve:
    zeros = np.zeros(N_INTERVALS, dtype=float)
    return DailySolve(
        q=zeros.copy(),
        charge=zeros.copy(),
        discharge=zeros.copy(),
        plan_unused=zeros.copy(),
        soc_end=np.full(N_INTERVALS, SOC_INITIAL_KWH),
        stage1_objective=0.0,
        stage2_objective=0.0,
        stage2_throughput=0.0,
        stage1_status="cold_start_no_forecast",
        stage2_status="cold_start_no_forecast",
        fallback_used=False,
    )


def fallback_plan(planned_net: np.ndarray, initial_soc: float) -> DailySolve:
    zeros = np.zeros(N_INTERVALS, dtype=float)
    return DailySolve(
        q=np.maximum(planned_net, 0.0),
        charge=zeros.copy(),
        discharge=zeros.copy(),
        plan_unused=np.maximum(-planned_net, 0.0),
        soc_end=np.full(N_INTERVALS, initial_soc),
        stage1_objective=float("nan"),
        stage2_objective=float("nan"),
        stage2_throughput=0.0,
        stage1_status="failed_fallback_activated",
        stage2_status="failed_fallback_activated",
        fallback_used=True,
    )


def execute_feedback_day(
    actual_net_kwh: np.ndarray,
    frozen_q_kwh: np.ndarray,
    initial_soc_kwh: float,
    enable_feedback: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    charge = np.zeros(N_INTERVALS, dtype=float)
    discharge = np.zeros(N_INTERVALS, dtype=float)
    emergency = np.zeros(N_INTERVALS, dtype=float)
    unused = np.zeros(N_INTERVALS, dtype=float)
    soc_start = np.zeros(N_INTERVALS, dtype=float)
    soc_end = np.zeros(N_INTERVALS, dtype=float)
    current_soc = float(initial_soc_kwh)

    for interval in range(N_INTERVALS):
        soc_start[interval] = current_soc
        residual = float(actual_net_kwh[interval] - frozen_q_kwh[interval])
        if enable_feedback:
            charge_max = min(
                INTERVAL_LIMIT_KWH,
                max(0.0, (SOC_MAX_KWH - current_soc) / ETA_CHARGE),
            )
            discharge_max = min(
                INTERVAL_LIMIT_KWH,
                max(0.0, ETA_DISCHARGE * (current_soc - SOC_MIN_KWH)),
            )
            charge[interval] = min(max(-residual, 0.0), charge_max)
            discharge[interval] = min(max(residual, 0.0), discharge_max)
        emergency[interval] = max(residual - discharge[interval], 0.0)
        unused[interval] = max(-residual - charge[interval], 0.0)
        current_soc = (
            current_soc
            + ETA_CHARGE * charge[interval]
            - discharge[interval] / ETA_DISCHARGE
        )
        soc_end[interval] = current_soc

    return charge, discharge, emergency, unused, soc_start, soc_end


def validate_day(
    actual_net: np.ndarray,
    planned_net: np.ndarray,
    initial_actual_soc: float,
    solve: DailySolve,
    actual_charge: np.ndarray,
    actual_discharge: np.ndarray,
    emergency: np.ndarray,
    actual_unused: np.ndarray,
    actual_soc_start: np.ndarray,
    actual_soc_end: np.ndarray,
) -> dict[str, float]:
    planned_soc_start = np.r_[initial_actual_soc, solve.soc_end[:-1]]
    planning_balance = (
        solve.q + solve.discharge - planned_net - solve.charge - solve.plan_unused
    )
    planning_storage = (
        solve.soc_end
        - planned_soc_start
        - ETA_CHARGE * solve.charge
        + solve.discharge / ETA_DISCHARGE
    )
    actual_balance = (
        solve.q
        + emergency
        + actual_discharge
        - actual_net
        - actual_charge
        - actual_unused
    )
    actual_storage = (
        actual_soc_end
        - actual_soc_start
        - ETA_CHARGE * actual_charge
        + actual_discharge / ETA_DISCHARGE
    )
    return {
        "planning_balance_max_abs": float(np.max(np.abs(planning_balance))),
        "planning_storage_recurrence_max_abs": float(np.max(np.abs(planning_storage))),
        "actual_balance_max_abs": float(np.max(np.abs(actual_balance))),
        "actual_storage_recurrence_max_abs": float(np.max(np.abs(actual_storage))),
        "planner_initial_soc_difference_abs": float(abs(planned_soc_start[0] - initial_actual_soc)),
        "actual_soc_chain_max_abs": float(
            np.max(np.abs(actual_soc_start[1:] - actual_soc_end[:-1]))
        ),
        "actual_soc_lower_violation": float(max(0.0, SOC_MIN_KWH - np.min(actual_soc_end))),
        "actual_soc_upper_violation": float(max(0.0, np.max(actual_soc_end) - SOC_MAX_KWH)),
        "actual_charge_limit_violation": float(
            max(0.0, np.max(actual_charge) - INTERVAL_LIMIT_KWH)
        ),
        "actual_discharge_limit_violation": float(
            max(0.0, np.max(actual_discharge) - INTERVAL_LIMIT_KWH)
        ),
        "actual_simultaneous_charge_discharge_max": float(
            np.max(np.minimum(actual_charge, actual_discharge))
        ),
        "actual_emergency_unused_overlap_max": float(
            np.max(np.minimum(emergency, actual_unused))
        ),
        "planned_simultaneous_charge_discharge_max": float(
            np.max(np.minimum(solve.charge, solve.discharge))
        ),
        "stage2_objective_excess": float(
            0.0
            if solve.fallback_used
            else max(0.0, solve.stage2_objective - solve.stage1_objective)
        ),
    }


def run_full_feedback_strategy(
    data: Q2Data,
    forecasts: ForecastBundle,
    margins: np.ndarray,
    config: StrategyConfig,
) -> StrategyRun:
    started = time.perf_counter()
    solver = DailyLP(data.price)
    actual_initial_soc = SOC_INITIAL_KWH
    ledger_rows: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []
    max_checks: dict[str, float] = {}
    fallback_days: list[str] = []

    for day_index in range(N_DAYS):
        day = data.dates[day_index]
        day_initial_soc = actual_initial_soc
        if day_index == 0:
            point_forecast = np.zeros(N_INTERVALS, dtype=float)
            margin = np.zeros(N_INTERVALS, dtype=float)
            planned_net = np.zeros(N_INTERVALS, dtype=float)
            solve = cold_start_plan()
            feedback_enabled = False
        else:
            point_forecast = forecasts.net_load_kwh[day_index]
            margin = (
                margins[day_index]
                if config.use_risk_margin
                else np.zeros(N_INTERVALS, dtype=float)
            )
            planned_net = point_forecast + margin
            terminal_value = (
                0.0
                if day_index == FINAL_DAY_INDEX
                else config.gamma * ETA_DISCHARGE * float(np.median(data.price))
            )
            try:
                solve = solver.solve(planned_net, day_initial_soc, terminal_value)
            except RuntimeError:
                solve = fallback_plan(planned_net, day_initial_soc)
                fallback_days.append(day.isoformat())
            feedback_enabled = True

        (
            actual_charge,
            actual_discharge,
            emergency,
            actual_unused,
            actual_soc_start,
            actual_soc_end,
        ) = execute_feedback_day(
            data.net_load_kwh[day_index],
            solve.q,
            day_initial_soc,
            feedback_enabled,
        )
        actual_terminal_soc = float(actual_soc_end[-1])
        contract_cost = float(data.price @ solve.q)
        emergency_cost = float(5.0 * data.price @ emergency)
        actual_cost = contract_cost + emergency_cost

        checks = validate_day(
            data.net_load_kwh[day_index],
            planned_net,
            day_initial_soc,
            solve,
            actual_charge,
            actual_discharge,
            emergency,
            actual_unused,
            actual_soc_start,
            actual_soc_end,
        )
        for name, value in checks.items():
            max_checks[name] = max(max_checks.get(name, 0.0), value)

        daily_rows.append(
            {
                "strategy_id": config.strategy_id,
                "date": day.isoformat(),
                "initial_soc_kwh": day_initial_soc,
                "planner_initial_soc_kwh": day_initial_soc,
                "planned_terminal_soc_kwh": float(solve.soc_end[-1]),
                "terminal_soc_kwh": actual_terminal_soc,
                "planned_purchase_kwh": float(solve.q.sum()),
                "planned_charge_kwh": float(solve.charge.sum()),
                "planned_discharge_kwh": float(solve.discharge.sum()),
                "charge_kwh": float(actual_charge.sum()),
                "discharge_kwh": float(actual_discharge.sum()),
                "planned_unused_kwh": float(solve.plan_unused.sum()),
                "emergency_purchase_kwh": float(emergency.sum()),
                "actual_unused_kwh": float(actual_unused.sum()),
                "contract_cost_yuan": contract_cost,
                "emergency_cost_yuan": emergency_cost,
                "actual_cost_yuan": actual_cost,
                "risk_margin_kwh": float(margin.sum()),
                "forecast_mae_kwh": (
                    float(np.mean(np.abs(data.net_load_kwh[day_index] - point_forecast)))
                    if day_index
                    else np.nan
                ),
                "stage1_objective": solve.stage1_objective,
                "stage2_objective": solve.stage2_objective,
                "stage2_throughput_kwh": solve.stage2_throughput,
                "fallback_used": solve.fallback_used,
            }
        )

        planned_soc_start = np.r_[day_initial_soc, solve.soc_end[:-1]]
        for interval in range(N_INTERVALS):
            ledger_rows.append(
                {
                    "strategy_id": config.strategy_id,
                    "date": day.isoformat(),
                    "interval_index": interval + 1,
                    "interval_start": data.interval_starts[interval],
                    "interval_end": data.interval_ends[interval],
                    "interval_label": data.interval_labels[interval],
                    "price_yuan_per_kwh": data.price[interval],
                    "actual_load_kw": data.load_kw[day_index, interval],
                    "actual_pv_kw": data.pv_kw[day_index, interval],
                    "actual_net_load_kwh": data.net_load_kwh[day_index, interval],
                    "forecast_load_kw": (
                        forecasts.load_kw[day_index, interval] if day_index else np.nan
                    ),
                    "forecast_pv_kw": (
                        forecasts.pv_kw[day_index, interval] if day_index else np.nan
                    ),
                    "point_forecast_net_kwh": (
                        point_forecast[interval] if day_index else np.nan
                    ),
                    "risk_margin_kwh": margin[interval],
                    "planned_net_load_kwh": planned_net[interval],
                    "planned_purchase_kwh": solve.q[interval],
                    "planned_charge_kwh": solve.charge[interval],
                    "planned_discharge_kwh": solve.discharge[interval],
                    "planned_unused_kwh": solve.plan_unused[interval],
                    "planned_soc_start_kwh": planned_soc_start[interval],
                    "planned_soc_end_kwh": solve.soc_end[interval],
                    "actual_charge_kwh": actual_charge[interval],
                    "actual_discharge_kwh": actual_discharge[interval],
                    "emergency_purchase_kwh": emergency[interval],
                    "actual_unused_kwh": actual_unused[interval],
                    "soc_start_kwh": actual_soc_start[interval],
                    "soc_end_kwh": actual_soc_end[interval],
                    "contract_cost_yuan": data.price[interval] * solve.q[interval],
                    "emergency_cost_yuan": 5.0 * data.price[interval] * emergency[interval],
                    "fallback_used": solve.fallback_used,
                }
            )

        actual_initial_soc = actual_terminal_soc

    ledger = pd.DataFrame(ledger_rows)
    daily = pd.DataFrame(daily_rows)
    evaluation_date = data.dates[EVALUATION_START_INDEX].isoformat()
    eval_ledger = ledger[ledger["date"] >= evaluation_date]
    eval_daily = daily[daily["date"] >= evaluation_date]
    emergency_values = eval_ledger["emergency_purchase_kwh"].to_numpy(dtype=float)
    emergency_total = float(emergency_values.sum())
    emergency_weighted_price = (
        float(
            eval_ledger["price_yuan_per_kwh"].to_numpy(dtype=float)
            @ emergency_values
            / emergency_total
        )
        if emergency_total > 0.0
        else 0.0
    )
    sorted_emergency = np.sort(emergency_values)[::-1]
    top_count = max(1, int(math.ceil(0.01 * len(sorted_emergency))))
    q_profiles = eval_ledger.pivot(
        index="date", columns="interval_index", values="planned_purchase_kwh"
    )
    unique_profiles = len({tuple(np.round(row, 6)) for row in q_profiles.to_numpy()})
    cross_day = (
        daily["initial_soc_kwh"].to_numpy(dtype=float)[1:]
        - daily["terminal_soc_kwh"].to_numpy(dtype=float)[:-1]
    )
    fee_residual = (
        daily["actual_cost_yuan"].to_numpy(dtype=float)
        - daily["contract_cost_yuan"].to_numpy(dtype=float)
        - daily["emergency_cost_yuan"].to_numpy(dtype=float)
    )
    next_day_planner_state = (
        daily["planner_initial_soc_kwh"].to_numpy(dtype=float)[1:]
        - daily["terminal_soc_kwh"].to_numpy(dtype=float)[:-1]
    )

    metrics = {
        "strategy_id": config.strategy_id,
        "role": config.role,
        "alpha": config.alpha,
        "gamma": config.gamma,
        "weight_mode": config.weight_mode,
        "execution_mode": "intraday_feedback_daily_replan_from_actual_soc",
        "evaluation_start": evaluation_date,
        "evaluation_end": data.dates[-1].isoformat(),
        "evaluation_days": int(len(eval_daily)),
        "contract_cost_yuan": float(eval_daily["contract_cost_yuan"].sum()),
        "emergency_cost_yuan": float(eval_daily["emergency_cost_yuan"].sum()),
        "actual_total_cost_yuan": float(eval_daily["actual_cost_yuan"].sum()),
        "planned_purchase_kwh": float(eval_daily["planned_purchase_kwh"].sum()),
        "emergency_purchase_kwh": emergency_total,
        "total_grid_purchase_kwh": float(
            eval_daily["planned_purchase_kwh"].sum() + emergency_total
        ),
        "actual_charge_kwh": float(eval_daily["charge_kwh"].sum()),
        "actual_discharge_kwh": float(eval_daily["discharge_kwh"].sum()),
        "planned_charge_kwh": float(eval_daily["planned_charge_kwh"].sum()),
        "planned_discharge_kwh": float(eval_daily["planned_discharge_kwh"].sum()),
        "planned_unused_kwh": float(eval_daily["planned_unused_kwh"].sum()),
        "actual_unused_kwh": float(eval_daily["actual_unused_kwh"].sum()),
        "soc_2025_02_01_start_kwh": float(eval_daily.iloc[0]["initial_soc_kwh"]),
        "soc_2025_12_31_end_kwh": float(eval_daily.iloc[-1]["terminal_soc_kwh"]),
        "forecast_mae_kwh": float(eval_daily["forecast_mae_kwh"].mean()),
        "emergency_interval_count": int(
            np.count_nonzero(emergency_values > NONZERO_TOLERANCE)
        ),
        "emergency_event_count": int(len(emergency_events(eval_ledger))),
        "emergency_weighted_base_price_yuan_per_kwh": emergency_weighted_price,
        "emergency_top_1pct_volume_share": float(
            sorted_emergency[:top_count].sum() / emergency_total
        )
        if emergency_total > 0.0
        else 0.0,
        "unique_daily_purchase_profiles_rounded_6dp": int(unique_profiles),
        "nonzero_actual_charge_interval_count": int(
            np.count_nonzero(
                eval_ledger["actual_charge_kwh"].to_numpy(dtype=float)
                > NONZERO_TOLERANCE
            )
        ),
        "nonzero_actual_discharge_interval_count": int(
            np.count_nonzero(
                eval_ledger["actual_discharge_kwh"].to_numpy(dtype=float)
                > NONZERO_TOLERANCE
            )
        ),
        "soc_at_lower_bound_interval_count": int(
            np.count_nonzero(
                np.isclose(
                    eval_ledger["soc_end_kwh"].to_numpy(dtype=float),
                    SOC_MIN_KWH,
                    atol=CHECK_TOLERANCE,
                )
            )
        ),
        "soc_at_upper_bound_interval_count": int(
            np.count_nonzero(
                np.isclose(
                    eval_ledger["soc_end_kwh"].to_numpy(dtype=float),
                    SOC_MAX_KWH,
                    atol=CHECK_TOLERANCE,
                )
            )
        ),
        "fallback_day_count": int(len(fallback_days)),
        "fallback_days": fallback_days,
        "january_actual_cost_yuan": float(
            daily[daily["date"] < evaluation_date]["actual_cost_yuan"].sum()
        ),
    }
    validation = {
        "strategy_id": config.strategy_id,
        "tolerance": CHECK_TOLERANCE,
        "max_residuals_or_violations": max_checks,
        "cross_day_actual_soc_continuity_max_abs": float(np.max(np.abs(cross_day))),
        "next_day_planner_uses_previous_actual_soc_max_abs": float(
            np.max(np.abs(next_day_planner_state))
        ),
        "daily_fee_reconciliation_max_abs": float(np.max(np.abs(fee_residual))),
        "daily_rows": int(len(daily)),
        "interval_rows": int(len(ledger)),
        "all_pass": False,
    }
    validation["all_pass"] = bool(
        not fallback_days
        and all(value <= CHECK_TOLERANCE for value in max_checks.values())
        and validation["cross_day_actual_soc_continuity_max_abs"] <= CHECK_TOLERANCE
        and validation["next_day_planner_uses_previous_actual_soc_max_abs"]
        <= CHECK_TOLERANCE
        and validation["daily_fee_reconciliation_max_abs"] <= CHECK_TOLERANCE
        and len(daily) == N_DAYS
        and len(ledger) == N_DAYS * N_INTERVALS
    )
    return StrategyRun(
        config=config,
        runtime_seconds=time.perf_counter() - started,
        ledger=ledger,
        daily=daily,
        metrics=metrics,
        validation=validation,
    )


def load_json_if_present(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def comparison_rows(main_run: StrategyRun, baseline_run: StrategyRun) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    old = load_json_if_present(
        PROJECT_ROOT
        / "results/Q2/experiments/round1/metrics/main_metrics.json"
    )
    replay = load_json_if_present(
        PROJECT_ROOT
        / "results/Q2/experiments/round2_same_contract_feedback/metrics/feedback_replay_metrics.json"
    )
    if old is not None:
        records.append(
            {
                "strategy": "round1_old_fixed_execution",
                "role": "legacy_comparator",
                "contract_cost_yuan": old["contract_cost_yuan"],
                "emergency_cost_yuan": old["emergency_cost_yuan"],
                "actual_total_cost_yuan": old["actual_total_cost_yuan"],
                "emergency_purchase_kwh": old["emergency_purchase_kwh"],
                "initial_soc_kwh": old["soc_2025_02_01_start_kwh"],
                "terminal_soc_kwh": old["soc_2025_12_31_end_kwh"],
            }
        )
    if replay is not None:
        records.append(
            {
                "strategy": "round2_same_contract_feedback",
                "role": "diagnostic",
                "contract_cost_yuan": replay["feedback_contract_cost_yuan"],
                "emergency_cost_yuan": replay["feedback_emergency_cost_yuan"],
                "actual_total_cost_yuan": replay["feedback_total_cost_yuan"],
                "emergency_purchase_kwh": replay["feedback_emergency_purchase_kwh"],
                "initial_soc_kwh": replay["initial_soc_kwh"],
                "terminal_soc_kwh": replay["feedback_terminal_soc_kwh"],
            }
        )
    for run in (main_run, baseline_run):
        records.append(
            {
                "strategy": run.config.strategy_id,
                "role": run.config.role,
                "contract_cost_yuan": run.metrics["contract_cost_yuan"],
                "emergency_cost_yuan": run.metrics["emergency_cost_yuan"],
                "actual_total_cost_yuan": run.metrics["actual_total_cost_yuan"],
                "emergency_purchase_kwh": run.metrics["emergency_purchase_kwh"],
                "initial_soc_kwh": run.metrics["soc_2025_02_01_start_kwh"],
                "terminal_soc_kwh": run.metrics["soc_2025_12_31_end_kwh"],
            }
        )
    comparison = pd.DataFrame(records)
    baseline_cost = float(baseline_run.metrics["actual_total_cost_yuan"])
    comparison["saving_vs_B2_yuan"] = baseline_cost - comparison["actual_total_cost_yuan"]
    comparison["saving_vs_B2_pct"] = (
        comparison["saving_vs_B2_yuan"] / baseline_cost * 100.0
    )
    return comparison


def main() -> None:
    started = time.perf_counter()
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    METRIC_DIR.mkdir(parents=True, exist_ok=True)
    data = load_q2_data()
    forecasts = build_forecasts(data, "exponential")
    margins = build_risk_margins(forecasts, 0.8)

    main_run = run_full_feedback_strategy(data, forecasts, margins, MAIN_CONFIG)
    baseline_run = run_full_feedback_strategy(data, forecasts, margins, BASELINE_CONFIG)
    comparison = comparison_rows(main_run, baseline_run)

    main_ledger_path = TABLE_DIR / "q2_M2_interval_ledger.csv"
    baseline_ledger_path = TABLE_DIR / "q2_B2_interval_ledger.csv"
    main_daily_path = TABLE_DIR / "q2_M2_daily_summary.csv"
    baseline_daily_path = TABLE_DIR / "q2_B2_daily_summary.csv"
    comparison_path = TABLE_DIR / "q2_full_feedback_comparison.csv"
    save_frame(main_run.ledger, main_ledger_path)
    save_frame(baseline_run.ledger, baseline_ledger_path)
    save_frame(main_run.daily, main_daily_path)
    save_frame(baseline_run.daily, baseline_daily_path)
    save_frame(comparison, comparison_path)

    evaluation_ledger = main_run.ledger[main_run.ledger["date"] >= "2025-02-01"]
    events_path = TABLE_DIR / "q2_M2_emergency_events.csv"
    storage_path = TABLE_DIR / "q2_M2_storage_4h.csv"
    save_frame(emergency_events(evaluation_ledger), events_path)
    save_frame(storage_four_hour_summary(evaluation_ledger), storage_path)
    table1, table2, table3 = specified_date_tables(main_run)
    table1_path = TABLE_DIR / "q2_M2_specified_date_table1.csv"
    table2_path = TABLE_DIR / "q2_M2_specified_date_table2.csv"
    table3_path = TABLE_DIR / "q2_M2_specified_date_table3.csv"
    save_frame(table1, table1_path)
    save_frame(table2, table2_path)
    save_frame(table3, table3_path)

    main_metrics_path = METRIC_DIR / "M2_main_metrics.json"
    baseline_metrics_path = METRIC_DIR / "B2_baseline_metrics.json"
    validation_path = METRIC_DIR / "validation_checks.json"
    write_json(main_metrics_path, main_run.metrics)
    write_json(baseline_metrics_path, baseline_run.metrics)
    validation = {
        "schema_version": 1,
        "main": main_run.validation,
        "baseline": baseline_run.validation,
        "direct_comparison": {
            "rules_differing": ["risk_margin_enabled"],
            "main_total_cost_yuan": main_run.metrics["actual_total_cost_yuan"],
            "baseline_total_cost_yuan": baseline_run.metrics["actual_total_cost_yuan"],
            "main_saving_yuan": (
                baseline_run.metrics["actual_total_cost_yuan"]
                - main_run.metrics["actual_total_cost_yuan"]
            ),
        },
        "all_pass": bool(
            main_run.validation["all_pass"] and baseline_run.validation["all_pass"]
        ),
    }
    write_json(validation_path, validation)

    output_paths = [
        main_ledger_path,
        baseline_ledger_path,
        main_daily_path,
        baseline_daily_path,
        comparison_path,
        events_path,
        storage_path,
        table1_path,
        table2_path,
        table3_path,
        main_metrics_path,
        baseline_metrics_path,
        validation_path,
    ]
    run_summary = {
        "schema_version": 1,
        "question": "Q2",
        "round": "round3_full_feedback",
        "approved_decision_id": "q2_method_choice_daily_replan_actual_soc_feedback",
        "methods": [
            {
                "method_id": "M2",
                "role": "main",
                "strategy_id": MAIN_CONFIG.strategy_id,
                "script": "code/Q2/q2_full_feedback.py",
                "status": "success" if main_run.validation["all_pass"] else "failed_validation",
                "execution_time_seconds": main_run.runtime_seconds,
                "metrics_summary": main_run.metrics,
            },
            {
                "method_id": "B2",
                "role": "usable_baseline",
                "strategy_id": BASELINE_CONFIG.strategy_id,
                "script": "code/Q2/q2_full_feedback.py",
                "status": "success" if baseline_run.validation["all_pass"] else "failed_validation",
                "execution_time_seconds": baseline_run.runtime_seconds,
                "metrics_summary": baseline_run.metrics,
            },
        ],
        "input_files": [
            {"path": str(PRICE_PATH), "sha256": sha256(PRICE_PATH)},
            {"path": str(ACTUAL_PATH), "sha256": sha256(ACTUAL_PATH)},
        ],
        "output_files": [
            str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
            for path in output_paths
        ],
        "comparison": validation["direct_comparison"],
        "output_degeneracy": {
            "M2_unique_daily_purchase_profiles": main_run.metrics[
                "unique_daily_purchase_profiles_rounded_6dp"
            ],
            "M2_emergency_intervals": main_run.metrics["emergency_interval_count"],
            "B2_unique_daily_purchase_profiles": baseline_run.metrics[
                "unique_daily_purchase_profiles_rounded_6dp"
            ],
            "B2_emergency_intervals": baseline_run.metrics["emergency_interval_count"],
        },
        "fallback_trigger": {
            "fallback_id": "F1",
            "condition": "unresolved daily stage-1 or stage-2 LP failure",
            "observed": bool(
                main_run.metrics["fallback_day_count"]
                or baseline_run.metrics["fallback_day_count"]
            ),
            "main_days": main_run.metrics["fallback_days"],
            "baseline_days": baseline_run.metrics["fallback_days"],
        },
        "warnings": [
            "M2 uses causal within-interval feedback but does not jointly optimize the feedback policy with the day-ahead LP.",
            "Round-1 formal workbook is not overwritten; a human result decision is required before replacement.",
        ],
        "errors": [],
        "environment": environment_info(),
        "random_seed": RANDOM_SEED,
        "validation_pass": validation["all_pass"],
        "execution_time_seconds": time.perf_counter() - started,
    }
    write_json(ROUND_DIR / "run_summary.json", run_summary)

    print(
        json.dumps(
            {
                "status": "success" if validation["all_pass"] else "failed_validation",
                "M2_contract_cost_yuan": main_run.metrics["contract_cost_yuan"],
                "M2_emergency_cost_yuan": main_run.metrics["emergency_cost_yuan"],
                "M2_total_cost_yuan": main_run.metrics["actual_total_cost_yuan"],
                "M2_emergency_purchase_kwh": main_run.metrics["emergency_purchase_kwh"],
                "M2_feb1_initial_soc_kwh": main_run.metrics["soc_2025_02_01_start_kwh"],
                "M2_year_end_soc_kwh": main_run.metrics["soc_2025_12_31_end_kwh"],
                "B2_total_cost_yuan": baseline_run.metrics["actual_total_cost_yuan"],
                "M2_saving_vs_B2_yuan": (
                    baseline_run.metrics["actual_total_cost_yuan"]
                    - main_run.metrics["actual_total_cost_yuan"]
                ),
                "validation_pass": validation["all_pass"],
                "runtime_seconds": run_summary["execution_time_seconds"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    raise SystemExit("请从辅助材料根目录运行 run.py，选择对应问题。")
