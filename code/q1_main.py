from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from q1_common import (
    COST_TOLERANCE_YUAN,
    ETA_CHARGE,
    ETA_DISCHARGE,
    INTERVAL_LIMIT_KWH,
    N_INTERVALS,
    SOC_INITIAL_KWH,
    SOC_MAX_KWH,
    SOC_MIN_KWH,
    Dispatch,
    Q1Input,
)
from scipy.optimize import linprog


@dataclass(frozen=True)
class MainSolveResult:
    dispatch: Dispatch
    stage1_cost: float
    stage2_cost: float
    stage2_throughput: float
    stage1_status: str
    stage2_status: str


def _build_linear_system(data: Q1Input) -> tuple[np.ndarray, np.ndarray, list[tuple[float | None, float | None]]]:
    n = N_INTERVALS
    a_eq = np.zeros((2 * n + 1, 5 * n), dtype=float)
    b_eq = np.zeros(2 * n + 1, dtype=float)

    for t in range(n):
        # q + pv + d = load + c + w
        a_eq[t, [t, n + t, 2 * n + t, 3 * n + t]] = [1.0, -1.0, 1.0, -1.0]
        b_eq[t] = data.load_kwh[t] - data.pv_kwh[t]

        # S_t = S_{t-1} + eta_c*c - d/eta_d
        a_eq[n + t, [n + t, 2 * n + t, 4 * n + t]] = [
            -ETA_CHARGE,
            1.0 / ETA_DISCHARGE,
            1.0,
        ]
        if t == 0:
            b_eq[n + t] = SOC_INITIAL_KWH
        else:
            a_eq[n + t, 4 * n + t - 1] = -1.0

    a_eq[-1, 5 * n - 1] = 1.0
    b_eq[-1] = SOC_INITIAL_KWH

    bounds: list[tuple[float | None, float | None]] = (
        [(0.0, None)] * n
        + [(0.0, INTERVAL_LIMIT_KWH)] * n
        + [(0.0, INTERVAL_LIMIT_KWH)] * n
        + [(0.0, None)] * n
        + [(SOC_MIN_KWH, SOC_MAX_KWH)] * n
    )
    return a_eq, b_eq, bounds


def solve_main(data: Q1Input) -> MainSolveResult:
    n = N_INTERVALS
    a_eq, b_eq, bounds = _build_linear_system(data)
    primary_objective = np.r_[data.price, np.zeros(4 * n)]

    stage1 = linprog(primary_objective, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not stage1.success:
        raise RuntimeError(f"Q1 stage-1 optimization failed: {stage1.message}")

    secondary_objective = np.r_[
        np.zeros(n),
        np.ones(n),
        np.ones(n),
        np.zeros(2 * n),
    ]
    stage2 = linprog(
        secondary_objective,
        A_ub=primary_objective.reshape(1, -1),
        b_ub=np.asarray([float(stage1.fun) + COST_TOLERANCE_YUAN]),
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )
    if not stage2.success:
        raise RuntimeError(f"Q1 stage-2 optimization failed: {stage2.message}")

    purchase, charge, discharge, unused, soc_end = stage2.x.reshape(5, n)
    stage2_cost = float(data.price @ purchase)
    dispatch = Dispatch(
        purchase=purchase,
        charge=charge,
        discharge=discharge,
        unused=unused,
        soc_end=soc_end,
        cost=stage2_cost,
    )
    return MainSolveResult(
        dispatch=dispatch,
        stage1_cost=float(stage1.fun),
        stage2_cost=stage2_cost,
        stage2_throughput=float(charge.sum() + discharge.sum()),
        stage1_status=str(stage1.message),
        stage2_status=str(stage2.message),
    )
