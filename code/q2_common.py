from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import sys
import time as time_module
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import numpy as np
import pandas as pd
import scipy
from openpyxl import load_workbook
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, lil_matrix, vstack


N_DAYS = 365
N_INTERVALS = 144
DELTA_HOURS = 1.0 / 6.0
ETA_CHARGE = 0.9
ETA_DISCHARGE = 0.9
SOC_INITIAL_KWH = 6000.0
SOC_MIN_KWH = 1200.0
SOC_MAX_KWH = 10800.0
POWER_LIMIT_KW = 5000.0
INTERVAL_LIMIT_KWH = POWER_LIMIT_KW * DELTA_HOURS
STAGE2_OBJECTIVE_TOLERANCE = 1e-6
CHECK_TOLERANCE = 2e-5
NONZERO_TOLERANCE = 1e-6
EVALUATION_START_INDEX = 31
FINAL_DAY_INDEX = 364
RANDOM_SEED = 2026

SOURCE_ROOT = PROJECT_ROOT / "data"
PRICE_PATH = SOURCE_ROOT / "附件1.xlsx"
ACTUAL_PATH = SOURCE_ROOT / "附件2.xlsx"
ROUND_DIR = PROJECT_ROOT / "results" / "Q2" / "experiments" / "round1"
TABLE_DIR = ROUND_DIR / "tables"
METRIC_DIR = ROUND_DIR / "metrics"
FIGURE_DIR = ROUND_DIR / "figures"


@dataclass(frozen=True)
class Q2Data:
    dates: list[date]
    source_labels: list[str]
    interval_starts: list[str]
    interval_ends: list[str]
    interval_labels: list[str]
    price: np.ndarray
    load_kw: np.ndarray
    pv_kw: np.ndarray
    net_load_kwh: np.ndarray
    price_sha256: str
    actual_sha256: str


@dataclass(frozen=True)
class ForecastBundle:
    load_kw: np.ndarray
    pv_kw: np.ndarray
    net_load_kwh: np.ndarray
    errors_kwh: np.ndarray
    valid: np.ndarray
    weight_mode: str


@dataclass(frozen=True)
class StrategyConfig:
    strategy_id: str
    role: str
    alpha: float | None
    gamma: float
    weight_mode: str
    use_risk_margin: bool


@dataclass(frozen=True)
class DailySolve:
    q: np.ndarray
    charge: np.ndarray
    discharge: np.ndarray
    plan_unused: np.ndarray
    soc_end: np.ndarray
    stage1_objective: float
    stage2_objective: float
    stage2_throughput: float
    stage1_status: str
    stage2_status: str
    fallback_used: bool


@dataclass
class StrategyRun:
    config: StrategyConfig
    runtime_seconds: float
    ledger: pd.DataFrame
    daily: pd.DataFrame
    metrics: dict[str, Any]
    validation: dict[str, Any]


def ensure_output_dirs() -> None:
    for path in (TABLE_DIR, METRIC_DIR, FIGURE_DIR):
        path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: Iterable[Iterable[Any]], header: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clock_label(value: Any) -> str:
    if isinstance(value, time):
        return f"{value.hour:02d}:{value.minute:02d}"
    return str(value).strip()


def _format_clock(total_minutes: int) -> str:
    if total_minutes == 24 * 60:
        return "24:00"
    hour, minute = divmod(total_minutes, 60)
    return f"{hour:02d}:{minute:02d}"


def _expected_end_label(index: int) -> set[str]:
    end_minutes = (index + 1) * 10
    if end_minutes == 24 * 60:
        return {"0:00+1", "00:00+1", "24:00"}
    label = _format_clock(end_minutes)
    return {label, label.lstrip("0")}


def _as_date(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    raise TypeError(f"Expected date value, got {value!r}")


def load_q2_data() -> Q2Data:
    for path in (PRICE_PATH, ACTUAL_PATH):
        if not path.is_file():
            raise FileNotFoundError(path)

    price_wb = load_workbook(PRICE_PATH, read_only=True, data_only=True)
    price_rows = list(price_wb.active.values)
    if tuple(price_rows[0]) != ("时间", "电价", "小区负载", "光伏发电预测功率"):
        raise ValueError(f"Unexpected attachment 1 headers: {price_rows[0]!r}")
    if len(price_rows) != N_INTERVALS + 1:
        raise ValueError(f"Expected 144 price rows, found {len(price_rows) - 1}")
    price_labels = [_clock_label(row[0]) for row in price_rows[1:]]
    prices = np.asarray([row[1] for row in price_rows[1:]], dtype=float)
    for index, label in enumerate(price_labels):
        if label not in _expected_end_label(index):
            raise ValueError(f"Unexpected price time label at interval {index}: {label!r}")
    if not np.isfinite(prices).all() or np.any(prices <= 0.0):
        raise ValueError("Prices must be finite and positive")

    actual_wb = load_workbook(ACTUAL_PATH, read_only=True, data_only=True)
    if actual_wb.sheetnames != ["小区负载", "光伏发电实际功率"]:
        raise ValueError(f"Unexpected attachment 2 sheets: {actual_wb.sheetnames}")

    matrices: dict[str, np.ndarray] = {}
    dates: list[date] | None = None
    source_labels: list[str] | None = None
    for sheet_name in actual_wb.sheetnames:
        sheet = actual_wb[sheet_name]
        rows = list(sheet.values)
        if len(rows) != N_DAYS + 1 or len(rows[0]) != N_INTERVALS + 1:
            raise ValueError(f"Unexpected shape for {sheet_name}: {len(rows)} x {len(rows[0])}")
        current_dates = [_as_date(row[0]) for row in rows[1:]]
        current_labels = [_clock_label(value) for value in rows[0][1:]]
        if dates is None:
            dates = current_dates
            source_labels = current_labels
        elif current_dates != dates or current_labels != source_labels:
            raise ValueError("Attachment 2 sheets do not share identical axes")
        matrices[sheet_name] = np.asarray([row[1:] for row in rows[1:]], dtype=float)

    assert dates is not None and source_labels is not None
    expected_dates = [date(2025, 1, 1) + timedelta(days=i) for i in range(N_DAYS)]
    if dates != expected_dates:
        raise ValueError("Attachment 2 dates are not the complete 2025 calendar")
    for index, label in enumerate(source_labels):
        if label not in _expected_end_label(index):
            raise ValueError(f"Unexpected actual time label at interval {index}: {label!r}")

    load_kw = matrices["小区负载"]
    pv_kw = matrices["光伏发电实际功率"]
    if not np.isfinite(load_kw).all() or not np.isfinite(pv_kw).all():
        raise ValueError("Attachment 2 contains missing or non-finite values")
    if np.any(load_kw < 0.0) or np.any(pv_kw < 0.0):
        raise ValueError("Load and PV must be nonnegative")

    starts = [_format_clock(i * 10) for i in range(N_INTERVALS)]
    ends = [_format_clock((i + 1) * 10) for i in range(N_INTERVALS)]
    labels = [f"{start}-{end}" for start, end in zip(starts, ends)]
    return Q2Data(
        dates=dates,
        source_labels=source_labels,
        interval_starts=starts,
        interval_ends=ends,
        interval_labels=labels,
        price=prices,
        load_kw=load_kw,
        pv_kw=pv_kw,
        net_load_kwh=(load_kw - pv_kw) * DELTA_HOURS,
        price_sha256=sha256(PRICE_PATH),
        actual_sha256=sha256(ACTUAL_PATH),
    )


def _is_weekend(value: date) -> bool:
    return value.weekday() >= 5


def _history_weights(day_index: int, indices: np.ndarray, weight_mode: str) -> np.ndarray:
    if weight_mode == "exponential":
        gaps = day_index - indices
        return np.exp(-gaps / 7.0)
    if weight_mode == "equal":
        return np.ones(len(indices), dtype=float)
    raise ValueError(f"Unknown weight mode: {weight_mode}")


def build_forecasts(data: Q2Data, weight_mode: str) -> ForecastBundle:
    predicted_load = np.full_like(data.load_kw, np.nan)
    predicted_pv = np.full_like(data.pv_kw, np.nan)
    predicted_net = np.full_like(data.net_load_kwh, np.nan)
    errors = np.full_like(data.net_load_kwh, np.nan)
    valid = np.zeros(N_DAYS, dtype=bool)

    for day_index in range(1, N_DAYS):
        start_28 = max(0, day_index - 28)
        prior_28 = np.arange(start_28, day_index, dtype=int)
        same_type = np.asarray(
            [idx for idx in prior_28 if _is_weekend(data.dates[idx]) == _is_weekend(data.dates[day_index])],
            dtype=int,
        )
        load_indices = same_type if len(same_type) else np.arange(max(0, day_index - 7), day_index, dtype=int)
        pv_indices = np.arange(max(0, day_index - 7), day_index, dtype=int)
        load_weights = _history_weights(day_index, load_indices, weight_mode)
        pv_weights = _history_weights(day_index, pv_indices, weight_mode)
        predicted_load[day_index] = np.average(data.load_kw[load_indices], axis=0, weights=load_weights)
        predicted_pv[day_index] = np.maximum(
            0.0,
            np.average(data.pv_kw[pv_indices], axis=0, weights=pv_weights),
        )
        predicted_net[day_index] = (predicted_load[day_index] - predicted_pv[day_index]) * DELTA_HOURS
        errors[day_index] = data.net_load_kwh[day_index] - predicted_net[day_index]
        valid[day_index] = True

    return ForecastBundle(
        load_kw=predicted_load,
        pv_kw=predicted_pv,
        net_load_kwh=predicted_net,
        errors_kwh=errors,
        valid=valid,
        weight_mode=weight_mode,
    )


def build_risk_margins(bundle: ForecastBundle, alpha: float) -> np.ndarray:
    margins = np.zeros((N_DAYS, N_INTERVALS), dtype=float)
    for day_index in range(1, N_DAYS):
        prior_indices = np.arange(max(1, day_index - 28), day_index, dtype=int)
        valid_indices = prior_indices[bundle.valid[prior_indices]]
        if len(np.unique(valid_indices)) < 7:
            continue
        for group in range(6):
            section = slice(group * 24, (group + 1) * 24)
            samples = bundle.errors_kwh[valid_indices, section].reshape(-1)
            quantile = float(np.quantile(samples, alpha, method="linear"))
            margins[day_index, section] = max(0.0, quantile)
    return margins


class DailyLP:
    def __init__(self, price: np.ndarray) -> None:
        self.price = np.asarray(price, dtype=float)
        n = N_INTERVALS
        a_eq = lil_matrix((2 * n, 5 * n), dtype=float)
        for t in range(n):
            a_eq[t, [t, n + t, 2 * n + t, 3 * n + t]] = [1.0, -1.0, 1.0, -1.0]
            a_eq[n + t, [n + t, 2 * n + t, 4 * n + t]] = [-ETA_CHARGE, 1.0 / ETA_DISCHARGE, 1.0]
            if t > 0:
                a_eq[n + t, 4 * n + t - 1] = -1.0
        self.a_eq = a_eq.tocsr()
        self.bounds = (
            [(0.0, None)] * n
            + [(0.0, INTERVAL_LIMIT_KWH)] * n
            + [(0.0, INTERVAL_LIMIT_KWH)] * n
            + [(0.0, None)] * n
            + [(SOC_MIN_KWH, SOC_MAX_KWH)] * n
        )
        self.secondary = np.r_[np.zeros(n), np.ones(n), np.ones(n), np.zeros(2 * n)]

    def solve(self, planned_net: np.ndarray, initial_soc: float, terminal_value: float) -> DailySolve:
        n = N_INTERVALS
        primary = np.r_[self.price, np.zeros(4 * n)]
        primary[5 * n - 1] = -terminal_value
        b_eq = np.r_[planned_net, np.zeros(n)]
        b_eq[n] = initial_soc

        stage1 = linprog(primary, A_eq=self.a_eq, b_eq=b_eq, bounds=self.bounds, method="highs")
        if not stage1.success:
            raise RuntimeError(f"Stage-1 LP failed: {stage1.message}")
        a_ub = csr_matrix(primary.reshape(1, -1))
        stage2 = linprog(
            self.secondary,
            A_ub=a_ub,
            b_ub=np.asarray([float(stage1.fun) + STAGE2_OBJECTIVE_TOLERANCE]),
            A_eq=self.a_eq,
            b_eq=b_eq,
            bounds=self.bounds,
            method="highs",
        )
        if not stage2.success:
            raise RuntimeError(f"Stage-2 LP failed: {stage2.message}")
        q, charge, discharge, plan_unused, soc_end = stage2.x.reshape(5, n)
        return DailySolve(
            q=q,
            charge=charge,
            discharge=discharge,
            plan_unused=plan_unused,
            soc_end=soc_end,
            stage1_objective=float(stage1.fun),
            stage2_objective=float(primary @ stage2.x),
            stage2_throughput=float(charge.sum() + discharge.sum()),
            stage1_status=str(stage1.message),
            stage2_status=str(stage2.message),
            fallback_used=False,
        )


def _cold_start(data: Q2Data) -> DailySolve:
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


def _daily_validation(
    data: Q2Data,
    day_index: int,
    initial_soc: float,
    planned_net: np.ndarray,
    point_forecast: np.ndarray,
    margin: np.ndarray,
    solve: DailySolve,
    emergency: np.ndarray,
    actual_unused: np.ndarray,
) -> dict[str, float]:
    previous_soc = np.r_[initial_soc, solve.soc_end[:-1]]
    planning_balance = solve.q + solve.discharge - planned_net - solve.charge - solve.plan_unused
    storage_recurrence = solve.soc_end - previous_soc - ETA_CHARGE * solve.charge + solve.discharge / ETA_DISCHARGE
    actual_balance = (
        solve.q
        + emergency
        + data.pv_kw[day_index] * DELTA_HOURS
        + solve.discharge
        - data.load_kw[day_index] * DELTA_HOURS
        - solve.charge
        - actual_unused
    )
    actual_residual = data.net_load_kwh[day_index] + solve.charge - solve.discharge - solve.q
    equation17 = actual_residual - (
        data.net_load_kwh[day_index] - point_forecast - margin - solve.plan_unused
    )
    return {
        "planning_balance_max_abs": float(np.max(np.abs(planning_balance))),
        "storage_recurrence_max_abs": float(np.max(np.abs(storage_recurrence))),
        "actual_balance_max_abs": float(np.max(np.abs(actual_balance))),
        "equation17_max_abs": float(np.max(np.abs(equation17))),
        "soc_lower_violation": float(max(0.0, SOC_MIN_KWH - np.min(solve.soc_end))),
        "soc_upper_violation": float(max(0.0, np.max(solve.soc_end) - SOC_MAX_KWH)),
        "charge_limit_violation": float(max(0.0, np.max(solve.charge) - INTERVAL_LIMIT_KWH)),
        "discharge_limit_violation": float(max(0.0, np.max(solve.discharge) - INTERVAL_LIMIT_KWH)),
        "simultaneous_charge_discharge_max": float(np.max(np.minimum(solve.charge, solve.discharge))),
        "emergency_unused_overlap_max": float(np.max(np.minimum(emergency, actual_unused))),
        "stage2_objective_excess": float(max(0.0, solve.stage2_objective - solve.stage1_objective)),
    }


def run_strategy(
    data: Q2Data,
    forecasts: ForecastBundle,
    margins: np.ndarray,
    config: StrategyConfig,
    start_day_index: int = 0,
    starting_soc_kwh: float = SOC_INITIAL_KWH,
) -> StrategyRun:
    started = time_module.perf_counter()
    solver = DailyLP(data.price)
    ledger_rows: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []
    if not 0 <= start_day_index < N_DAYS:
        raise ValueError(f"Invalid start_day_index: {start_day_index}")
    initial_soc = float(starting_soc_kwh)
    max_checks: dict[str, float] = {}
    fallback_days: list[str] = []

    for day_index in range(start_day_index, N_DAYS):
        day = data.dates[day_index]
        if day_index == 0:
            point_forecast = np.zeros(N_INTERVALS, dtype=float)
            margin = np.zeros(N_INTERVALS, dtype=float)
            planned_net = np.zeros(N_INTERVALS, dtype=float)
            solve = _cold_start(data)
        else:
            point_forecast = forecasts.net_load_kwh[day_index]
            margin = margins[day_index] if config.use_risk_margin else np.zeros(N_INTERVALS, dtype=float)
            planned_net = point_forecast + margin
            terminal_value = 0.0 if day_index == FINAL_DAY_INDEX else config.gamma * ETA_DISCHARGE * float(np.median(data.price))
            try:
                solve = solver.solve(planned_net, initial_soc, terminal_value)
            except RuntimeError:
                q = np.maximum(planned_net, 0.0)
                zeros = np.zeros(N_INTERVALS, dtype=float)
                solve = DailySolve(
                    q=q,
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
                fallback_days.append(str(day))

        actual_residual = data.net_load_kwh[day_index] + solve.charge - solve.discharge - solve.q
        emergency = np.maximum(actual_residual, 0.0)
        actual_unused = np.maximum(-actual_residual, 0.0)
        contract_cost = float(data.price @ solve.q)
        emergency_cost = float(5.0 * data.price @ emergency)
        actual_cost = contract_cost + emergency_cost
        checks = _daily_validation(
            data,
            day_index,
            initial_soc,
            planned_net,
            point_forecast,
            margin,
            solve,
            emergency,
            actual_unused,
        )
        for name, value in checks.items():
            max_checks[name] = max(max_checks.get(name, 0.0), value)

        daily_rows.append(
            {
                "strategy_id": config.strategy_id,
                "date": day.isoformat(),
                "initial_soc_kwh": initial_soc,
                "terminal_soc_kwh": float(solve.soc_end[-1]),
                "planned_purchase_kwh": float(solve.q.sum()),
                "charge_kwh": float(solve.charge.sum()),
                "discharge_kwh": float(solve.discharge.sum()),
                "planned_unused_kwh": float(solve.plan_unused.sum()),
                "emergency_purchase_kwh": float(emergency.sum()),
                "actual_unused_kwh": float(actual_unused.sum()),
                "contract_cost_yuan": contract_cost,
                "emergency_cost_yuan": emergency_cost,
                "actual_cost_yuan": actual_cost,
                "risk_margin_kwh": float(margin.sum()),
                "forecast_mae_kwh": float(np.mean(np.abs(data.net_load_kwh[day_index] - point_forecast))) if day_index else np.nan,
                "stage1_objective": solve.stage1_objective,
                "stage2_objective": solve.stage2_objective,
                "stage2_throughput_kwh": solve.stage2_throughput,
                "fallback_used": solve.fallback_used,
            }
        )

        previous_soc = np.r_[initial_soc, solve.soc_end[:-1]]
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
                    "forecast_load_kw": forecasts.load_kw[day_index, interval] if day_index else np.nan,
                    "forecast_pv_kw": forecasts.pv_kw[day_index, interval] if day_index else np.nan,
                    "point_forecast_net_kwh": point_forecast[interval] if day_index else np.nan,
                    "risk_margin_kwh": margin[interval],
                    "planned_net_load_kwh": planned_net[interval],
                    "planned_purchase_kwh": solve.q[interval],
                    "actual_charge_kwh": solve.charge[interval],
                    "actual_discharge_kwh": solve.discharge[interval],
                    "planned_unused_kwh": solve.plan_unused[interval],
                    "emergency_purchase_kwh": emergency[interval],
                    "actual_unused_kwh": actual_unused[interval],
                    "soc_start_kwh": previous_soc[interval],
                    "soc_end_kwh": solve.soc_end[interval],
                    "contract_cost_yuan": data.price[interval] * solve.q[interval],
                    "emergency_cost_yuan": 5.0 * data.price[interval] * emergency[interval],
                    "fallback_used": solve.fallback_used,
                }
            )
        initial_soc = float(solve.soc_end[-1])

    ledger = pd.DataFrame(ledger_rows)
    daily = pd.DataFrame(daily_rows)
    eval_ledger = ledger[ledger["date"] >= data.dates[EVALUATION_START_INDEX].isoformat()]
    eval_daily = daily[daily["date"] >= data.dates[EVALUATION_START_INDEX].isoformat()]
    q_profiles = eval_ledger.pivot(index="date", columns="interval_index", values="planned_purchase_kwh")
    unique_profiles = len({tuple(np.round(row, 6)) for row in q_profiles.to_numpy()})
    emergency_values = eval_ledger["emergency_purchase_kwh"].to_numpy()
    sorted_emergency = np.sort(emergency_values)[::-1]
    top_count = max(1, int(math.ceil(0.01 * len(sorted_emergency))))
    total_emergency = float(sorted_emergency.sum())
    top_one_percent_share = float(sorted_emergency[:top_count].sum() / total_emergency) if total_emergency > 0 else 0.0

    metrics = {
        "strategy_id": config.strategy_id,
        "role": config.role,
        "alpha": config.alpha,
        "gamma": config.gamma,
        "weight_mode": config.weight_mode,
        "evaluation_start": data.dates[EVALUATION_START_INDEX].isoformat(),
        "evaluation_end": data.dates[-1].isoformat(),
        "evaluation_days": int(len(eval_daily)),
        "contract_cost_yuan": float(eval_daily["contract_cost_yuan"].sum()),
        "emergency_cost_yuan": float(eval_daily["emergency_cost_yuan"].sum()),
        "actual_total_cost_yuan": float(eval_daily["actual_cost_yuan"].sum()),
        "planned_purchase_kwh": float(eval_daily["planned_purchase_kwh"].sum()),
        "emergency_purchase_kwh": float(eval_daily["emergency_purchase_kwh"].sum()),
        "total_grid_purchase_kwh": float(eval_daily["planned_purchase_kwh"].sum() + eval_daily["emergency_purchase_kwh"].sum()),
        "charge_kwh": float(eval_daily["charge_kwh"].sum()),
        "discharge_kwh": float(eval_daily["discharge_kwh"].sum()),
        "planned_unused_kwh": float(eval_daily["planned_unused_kwh"].sum()),
        "actual_unused_kwh": float(eval_daily["actual_unused_kwh"].sum()),
        "soc_2025_02_01_start_kwh": float(eval_daily.iloc[0]["initial_soc_kwh"]),
        "soc_2025_12_31_end_kwh": float(eval_daily.iloc[-1]["terminal_soc_kwh"]),
        "forecast_mae_kwh": float(eval_daily["forecast_mae_kwh"].mean()),
        "emergency_interval_count": int(np.count_nonzero(emergency_values > NONZERO_TOLERANCE)),
        "nonzero_charge_interval_count": int(np.count_nonzero(eval_ledger["actual_charge_kwh"].to_numpy() > NONZERO_TOLERANCE)),
        "nonzero_discharge_interval_count": int(np.count_nonzero(eval_ledger["actual_discharge_kwh"].to_numpy() > NONZERO_TOLERANCE)),
        "unique_daily_purchase_profiles_rounded_6dp": int(unique_profiles),
        "emergency_top_1pct_volume_share": top_one_percent_share,
        "charge_at_limit_interval_count": int(np.count_nonzero(np.isclose(eval_ledger["actual_charge_kwh"], INTERVAL_LIMIT_KWH, atol=CHECK_TOLERANCE))),
        "discharge_at_limit_interval_count": int(np.count_nonzero(np.isclose(eval_ledger["actual_discharge_kwh"], INTERVAL_LIMIT_KWH, atol=CHECK_TOLERANCE))),
        "soc_at_lower_bound_interval_count": int(np.count_nonzero(np.isclose(eval_ledger["soc_end_kwh"], SOC_MIN_KWH, atol=CHECK_TOLERANCE))),
        "soc_at_upper_bound_interval_count": int(np.count_nonzero(np.isclose(eval_ledger["soc_end_kwh"], SOC_MAX_KWH, atol=CHECK_TOLERANCE))),
        "fallback_day_count": int(len(fallback_days)),
        "fallback_days": fallback_days,
        "january_actual_cost_yuan": float(
            daily[daily["date"] < data.dates[EVALUATION_START_INDEX].isoformat()]["actual_cost_yuan"].sum()
        ),
        "run_start_date": data.dates[start_day_index].isoformat(),
        "starting_soc_kwh": float(starting_soc_kwh),
    }
    validation = {
        "strategy_id": config.strategy_id,
        "tolerance": CHECK_TOLERANCE,
        "max_residuals_or_violations": max_checks,
        "all_pass": bool(
            not fallback_days
            and all(value <= CHECK_TOLERANCE for value in max_checks.values())
            and len(daily) == N_DAYS - start_day_index
            and len(ledger) == (N_DAYS - start_day_index) * N_INTERVALS
        ),
        "daily_rows": int(len(daily)),
        "interval_rows": int(len(ledger)),
        "cross_day_continuity_max_abs": float(
            np.max(
                np.abs(
                    daily["initial_soc_kwh"].to_numpy()[1:]
                    - daily["terminal_soc_kwh"].to_numpy()[:-1]
                )
            )
        ),
        "cost_reconciliation_max_abs": float(
            np.max(
                np.abs(
                    daily["actual_cost_yuan"].to_numpy()
                    - daily["contract_cost_yuan"].to_numpy()
                    - daily["emergency_cost_yuan"].to_numpy()
                )
            )
        ),
    }
    validation["all_pass"] = bool(
        validation["all_pass"]
        and validation["cross_day_continuity_max_abs"] <= CHECK_TOLERANCE
        and validation["cost_reconciliation_max_abs"] <= CHECK_TOLERANCE
    )
    return StrategyRun(
        config=config,
        runtime_seconds=time_module.perf_counter() - started,
        ledger=ledger,
        daily=daily,
        metrics=metrics,
        validation=validation,
    )


def emergency_events(ledger: pd.DataFrame) -> pd.DataFrame:
    events: list[dict[str, Any]] = []
    for day, group in ledger.groupby("date", sort=True):
        ordered = group.sort_values("interval_index")
        active = ordered["emergency_purchase_kwh"].to_numpy() > NONZERO_TOLERANCE
        amounts = ordered["emergency_purchase_kwh"].to_numpy()
        starts = ordered["interval_start"].tolist()
        ends = ordered["interval_end"].tolist()
        cursor = 0
        event_index = 0
        while cursor < N_INTERVALS:
            if not active[cursor]:
                cursor += 1
                continue
            event_start = cursor
            while cursor + 1 < N_INTERVALS and active[cursor + 1]:
                cursor += 1
            event_end = cursor
            event_index += 1
            events.append(
                {
                    "date": day,
                    "event_index": event_index,
                    "start_interval_index": event_start + 1,
                    "end_interval_index": event_end + 1,
                    "emergency_time_range": f"{starts[event_start]}-{ends[event_end]}",
                    "emergency_purchase_kwh": float(amounts[event_start : event_end + 1].sum()),
                }
            )
            cursor += 1
    return pd.DataFrame(events)


def storage_four_hour_summary(ledger: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for day, group in ledger.groupby("date", sort=True):
        ordered = group.sort_values("interval_index")
        initial_soc = float(ordered.iloc[0]["soc_start_kwh"])
        terminal_soc = float(ordered.iloc[-1]["soc_end_kwh"])
        for block in range(6):
            section = ordered.iloc[block * 24 : (block + 1) * 24]
            rows.append(
                {
                    "date": day,
                    "block_index": block + 1,
                    "time_range": f"{block * 4:02d}:00-{(block + 1) * 4:02d}:00" if block < 5 else "20:00-24:00",
                    "charge_kwh": float(section["actual_charge_kwh"].sum()),
                    "discharge_kwh": float(section["actual_discharge_kwh"].sum()),
                    "soc_00_kwh": initial_soc if block == 0 else np.nan,
                    "soc_24_kwh": terminal_soc if block == 1 else np.nan,
                }
            )
    return pd.DataFrame(rows)


def specified_date_tables(run: StrategyRun) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    specified = ["2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21"]
    interval_indices = [61, 73, 85, 97, 109, 121]
    table1_rows: list[dict[str, Any]] = []
    for day in specified:
        ledger_day = run.ledger[run.ledger["date"] == day].set_index("interval_index")
        daily = run.daily[run.daily["date"] == day].iloc[0]
        row: dict[str, Any] = {"date": day}
        for interval_index in interval_indices:
            label = ledger_day.loc[interval_index, "interval_label"]
            row[f"{label}_purchase_kwh"] = float(ledger_day.loc[interval_index, "planned_purchase_kwh"])
        row["daily_planned_purchase_kwh"] = float(daily["planned_purchase_kwh"])
        row["daily_contract_cost_yuan"] = float(daily["contract_cost_yuan"])
        row["daily_emergency_cost_yuan"] = float(daily["emergency_cost_yuan"])
        row["daily_actual_total_cost_yuan"] = float(daily["actual_cost_yuan"])
        table1_rows.append(row)
    table1 = pd.DataFrame(table1_rows)
    table2 = storage_four_hour_summary(run.ledger[run.ledger["date"].isin(specified)])
    table3 = emergency_events(run.ledger[run.ledger["date"].isin(specified)])
    return table1, table2, table3


def perfect_foresight_lower_bound(data: Q2Data, initial_soc: float) -> dict[str, Any]:
    started = time_module.perf_counter()
    net = data.net_load_kwh[EVALUATION_START_INDEX:].reshape(-1)
    price = np.tile(data.price, N_DAYS - EVALUATION_START_INDEX)
    n = len(net)
    a_eq = lil_matrix((2 * n, 5 * n), dtype=float)
    for t in range(n):
        a_eq[t, [t, n + t, 2 * n + t, 3 * n + t]] = [1.0, -1.0, 1.0, -1.0]
        a_eq[n + t, [n + t, 2 * n + t, 4 * n + t]] = [-ETA_CHARGE, 1.0 / ETA_DISCHARGE, 1.0]
        if t > 0:
            a_eq[n + t, 4 * n + t - 1] = -1.0
    b_eq = np.r_[net, np.zeros(n)]
    b_eq[n] = initial_soc
    bounds = (
        [(0.0, None)] * n
        + [(0.0, INTERVAL_LIMIT_KWH)] * n
        + [(0.0, INTERVAL_LIMIT_KWH)] * n
        + [(0.0, None)] * n
        + [(SOC_MIN_KWH, SOC_MAX_KWH)] * n
    )
    objective = np.r_[price, np.zeros(4 * n)]
    result = linprog(objective, A_eq=a_eq.tocsr(), b_eq=b_eq, bounds=bounds, method="highs")
    if not result.success:
        raise RuntimeError(f"Perfect-foresight lower bound failed: {result.message}")
    q, charge, discharge, unused, soc_end = result.x.reshape(5, n)
    balance_residual = q + discharge - net - charge - unused
    previous_soc = np.r_[initial_soc, soc_end[:-1]]
    storage_residual = soc_end - previous_soc - ETA_CHARGE * charge + discharge / ETA_DISCHARGE
    return {
        "evaluation_start": data.dates[EVALUATION_START_INDEX].isoformat(),
        "evaluation_end": data.dates[-1].isoformat(),
        "initial_soc_kwh": float(initial_soc),
        "lower_bound_cost_yuan": float(result.fun),
        "planned_purchase_kwh": float(q.sum()),
        "terminal_soc_kwh": float(soc_end[-1]),
        "balance_max_abs": float(np.max(np.abs(balance_residual))),
        "storage_recurrence_max_abs": float(np.max(np.abs(storage_residual))),
        "solver_status": str(result.message),
        "runtime_seconds": time_module.perf_counter() - started,
    }


def environment_info() -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "solver": "scipy.optimize.linprog(method=highs)",
        "random_seed": RANDOM_SEED,
    }
