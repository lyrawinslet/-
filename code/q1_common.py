from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from datetime import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import numpy as np
from openpyxl import load_workbook

N_INTERVALS = 144
DELTA_HOURS = 1.0 / 6.0
ETA_CHARGE = 0.9
ETA_DISCHARGE = 0.9
SOC_INITIAL_KWH = 6000.0
SOC_MIN_KWH = 1200.0
SOC_MAX_KWH = 10800.0
POWER_LIMIT_KW = 5000.0
INTERVAL_LIMIT_KWH = POWER_LIMIT_KW * DELTA_HOURS
COST_TOLERANCE_YUAN = 1e-6
CHECK_TOLERANCE = 1e-7

SOURCE_PATH = PROJECT_ROOT / "data" / "附件1.xlsx"
TEMPLATE_PATH = PROJECT_ROOT / "result1.xlsx"
ROUND_DIR = PROJECT_ROOT / "results" / "Q1" / "experiments" / "round1"
TABLE_DIR = ROUND_DIR / "tables"
METRIC_DIR = ROUND_DIR / "metrics"


@dataclass(frozen=True)
class Q1Input:
    source_labels: list[str]
    interval_starts: list[str]
    interval_ends: list[str]
    interval_labels: list[str]
    price: np.ndarray
    load_kw: np.ndarray
    pv_kw: np.ndarray
    load_kwh: np.ndarray
    pv_kwh: np.ndarray
    source_sha256: str


@dataclass(frozen=True)
class Dispatch:
    purchase: np.ndarray
    charge: np.ndarray
    discharge: np.ndarray
    unused: np.ndarray
    soc_end: np.ndarray
    cost: float


def _format_clock(total_minutes: int) -> str:
    if total_minutes == 24 * 60:
        return "24:00"
    hour, minute = divmod(total_minutes, 60)
    return f"{hour:02d}:{minute:02d}"


def _source_label(value: Any) -> str:
    if isinstance(value, time):
        return f"{value.hour:02d}:{value.minute:02d}"
    return str(value).strip()


def _expected_source_label(index: int) -> set[str]:
    end_minutes = (index + 1) * 10
    if end_minutes == 24 * 60:
        return {"0:00+1", "00:00+1", "24:00"}
    return {_format_clock(end_minutes), _format_clock(end_minutes).lstrip("0")}


def load_q1_input(path: Path = SOURCE_PATH) -> Q1Input:
    if not path.is_file():
        raise FileNotFoundError(f"Required Q1 input not found: {path}")
    workbook = load_workbook(path, read_only=True, data_only=True)
    rows = list(workbook.active.values)
    expected_headers = ("时间", "电价", "小区负载", "光伏发电预测功率")
    if tuple(rows[0]) != expected_headers:
        raise ValueError(f"Unexpected Q1 headers: {rows[0]!r}")
    data_rows = rows[1:]
    if len(data_rows) != N_INTERVALS:
        raise ValueError(f"Expected 144 Q1 intervals, found {len(data_rows)}")

    source_labels = [_source_label(row[0]) for row in data_rows]
    for index, label in enumerate(source_labels):
        if label not in _expected_source_label(index):
            raise ValueError(
                f"Unexpected source time label at row {index + 2}: {label!r}; "
                f"expected one of {_expected_source_label(index)}"
            )

    numeric = np.asarray([row[1:] for row in data_rows], dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("Q1 input contains missing or non-finite numeric values")
    price, load_kw, pv_kw = numeric.T
    if np.any(price <= 0.0) or np.any(load_kw < 0.0) or np.any(pv_kw < 0.0):
        raise ValueError("Q1 input violates price/load/PV sign requirements")

    starts = [_format_clock(i * 10) for i in range(N_INTERVALS)]
    ends = [_format_clock((i + 1) * 10) for i in range(N_INTERVALS)]
    labels = [f"{start}-{end}" for start, end in zip(starts, ends)]
    return Q1Input(
        source_labels=source_labels,
        interval_starts=starts,
        interval_ends=ends,
        interval_labels=labels,
        price=price,
        load_kw=load_kw,
        pv_kw=pv_kw,
        load_kwh=load_kw * DELTA_HOURS,
        pv_kwh=pv_kw * DELTA_HOURS,
        source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def baseline_dispatch(data: Q1Input) -> Dispatch:
    purchase = np.maximum(data.load_kwh - data.pv_kwh, 0.0)
    unused = np.maximum(data.pv_kwh - data.load_kwh, 0.0)
    zeros = np.zeros(N_INTERVALS)
    soc_end = np.full(N_INTERVALS, SOC_INITIAL_KWH)
    return Dispatch(
        purchase=purchase,
        charge=zeros.copy(),
        discharge=zeros.copy(),
        unused=unused,
        soc_end=soc_end,
        cost=float(data.price @ purchase),
    )


def ensure_output_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    METRIC_DIR.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def finite_float(value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"Non-finite output: {value}")
    return value
