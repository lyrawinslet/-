from __future__ import annotations

from q1_common import Dispatch, Q1Input, baseline_dispatch


def solve_baseline(data: Q1Input) -> Dispatch:
    """Compute the approved no-storage baseline."""
    return baseline_dispatch(data)
