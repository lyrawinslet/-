from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from q4_model import (
    EVALUATION_START_INDEX,
    FORMAL_PRICE_INFORMATION,
    N_DAYS,
    PROJECT_ROOT,
    PolicyConfig,
    build_forecast_archive,
    build_inherited_risk_margins,
    load_data,
    run_policy,
    write_json,
)


ROUND_DIR = PROJECT_ROOT / "results" / "Q4" / "experiments" / "round2_inherited_price"
OUTPUT_PATH = PROJECT_ROOT / "robustness" / "Q4" / "q4_robustness_summary.json"


def load_formal_metrics(name: str) -> dict[str, Any]:
    path = ROUND_DIR / "metrics" / name
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def comparison(main: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float]:
    main_cost = float(main["actual_total_cost_yuan"])
    baseline_cost = float(baseline["actual_total_cost_yuan"])
    main_emergency = float(main["emergency_purchase_kwh"])
    baseline_emergency = float(baseline["emergency_purchase_kwh"])
    return {
        "main_minus_baseline_cost_yuan": main_cost - baseline_cost,
        "main_cost_change_vs_baseline_pct": (main_cost / baseline_cost - 1.0) * 100.0,
        "cost_saving_vs_baseline_yuan": baseline_cost - main_cost,
        "cost_saving_vs_baseline_pct": (1.0 - main_cost / baseline_cost) * 100.0,
        "main_minus_baseline_emergency_kwh": main_emergency - baseline_emergency,
        "emergency_reduction_vs_baseline_kwh": baseline_emergency - main_emergency,
        "emergency_reduction_vs_baseline_pct": (1.0 - main_emergency / baseline_emergency) * 100.0,
    }


def selected_metrics(run: Any) -> dict[str, Any]:
    keys = (
        "strategy_id",
        "role",
        "question_variant",
        "alpha",
        "simulation_start",
        "simulation_days",
        "run_initial_soc_kwh",
        "terminal_soc_evaluation_kwh",
        "original_contract_kwh",
        "final_contract_kwh",
        "adjustment_increase_kwh",
        "adjustment_decrease_kwh",
        "adjustment_transaction_count",
        "contract_total_cost_yuan",
        "emergency_cost_yuan",
        "emergency_purchase_kwh",
        "actual_total_cost_yuan",
        "planning_calls_full_run",
        "degenerate_scenario_calls_full_run",
        "fallback_event_count",
    )
    return {
        **{key: run.metrics[key] for key in keys},
        "runtime_seconds": run.runtime_seconds,
        "hard_checks_pass": run.validation["hard_checks_pass"],
        "maximum_reported_validation_residual": max(
            abs(float(value))
            for key, value in run.validation.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool) and key != "incremental_example_100_to_80_to_100_yuan"
        ),
    }


def main() -> None:
    data = load_data()
    archives = {
        "Q4-2": build_forecast_archive(data, "historical"),
        "Q4-3": build_forecast_archive(data, "official"),
    }
    formal = {
        "Q4-2": {
            "main": load_formal_metrics("q42_main_metrics.json"),
            "baseline": load_formal_metrics("q42_baseline_metrics.json"),
        },
        "Q4-3": {
            "main": load_formal_metrics("q43_main_metrics.json"),
            "baseline": load_formal_metrics("q43_baseline_metrics.json"),
        },
    }
    configs = {
        "Q4-2": (
            PolicyConfig("M4I_Q42_common_soc_alpha08", "main", "Q4-2", "historical", "inherited", False, alpha=0.8),
            PolicyConfig("B4I_Q42_common_soc_b0", "usable_baseline", "Q4-2", "historical", "inherited", False, alpha=None),
        ),
        "Q4-3": (
            PolicyConfig("M4I_Q43_common_soc_alpha08", "main", "Q4-3", "official", "inherited", True, alpha=0.8),
            PolicyConfig("B4I_Q43_common_soc_b0", "usable_baseline", "Q4-3", "official", "inherited", True, alpha=None),
        ),
    }

    variants: dict[str, Any] = {}
    all_checks_pass = True
    for variant in ("Q4-2", "Q4-3"):
        common_soc = float(formal[variant]["main"]["initial_soc_evaluation_kwh"])
        runs = []
        for config in configs[variant]:
            print(f"START {config.strategy_id} common_soc={common_soc:.12f}", flush=True)
            margins = build_inherited_risk_margins(archives[variant], variant, config.alpha)
            run = run_policy(
                data,
                archives[variant],
                config,
                margins=margins,
                max_days=N_DAYS,
                start_day_index=EVALUATION_START_INDEX,
                initial_soc_override=common_soc,
            )
            print(
                f"DONE {config.strategy_id} cost={run.metrics['actual_total_cost_yuan']:.2f} "
                f"emergency={run.metrics['emergency_purchase_kwh']:.3f} "
                f"terminal_soc={run.metrics['terminal_soc_evaluation_kwh']:.3f} "
                f"hard_checks={run.validation['hard_checks_pass']}",
                flush=True,
            )
            all_checks_pass = all_checks_pass and bool(run.validation["hard_checks_pass"])
            runs.append(run)

        paired = comparison(runs[0].metrics, runs[1].metrics)
        formal_comparison = comparison(formal[variant]["main"], formal[variant]["baseline"])
        variants[variant] = {
            "common_feb01_soc_kwh": common_soc,
            "common_soc_selection_rule": "use the formal main policy's February 1 initial SOC for this subquestion",
            "main": selected_metrics(runs[0]),
            "baseline": selected_metrics(runs[1]),
            "paired_common_soc_comparison": paired,
            "formal_end_to_end_whole_policy_comparison": formal_comparison,
            "interpretation": (
                "The paired replay starts both policies from the same February 1 SOC and therefore isolates the "
                "evaluation-period risk-margin difference conditional on that selected SOC. The formal comparison also "
                "contains the policies' different January state trajectories and is not an alpha-only estimate."
            ),
        }

    payload = {
        "schema_version": 1,
        "question": "Q4",
        "test_id": "q4_round2_common_feb01_soc_paired_replay_v1",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "PASS" if all_checks_pass else "FAIL",
        "claim_scope": "exploratory paired replay; formal runs and workbooks are unchanged",
        "price_information": FORMAL_PRICE_INFORMATION,
        "fixed_elements": [
            "pre-existing historical forecast calibration and official forecast archive",
            "attachment 4 target-day full 144-interval price path",
            "forecast, feedback, storage, settlement, and all other Q4 round2 rules",
        ],
        "changed_element": "both main and corresponding zero-margin baseline start at the same February 1 SOC",
        "common_soc_choice_limit": (
            "Each subquestion uses its formal main policy's February 1 SOC as the common state. "
            "This is a transparent conditional comparison, not proof that the result is invariant to every feasible common SOC."
        ),
        "variants": variants,
        "input_files": ["data/附件2.xlsx", "data/附件3.xlsx", "data/附件4.xlsx"],
        "source_files": [f"metrics/{name}" for name in (
            "q42_main_metrics.json", "q42_baseline_metrics.json",
            "q43_main_metrics.json", "q43_baseline_metrics.json")],
        "source_files_base": "本JSON文件所在的本次Q4计算输出目录",
    }
    write_json(OUTPUT_PATH, payload)
    print(f"WROTE {OUTPUT_PATH}", flush=True)


if __name__ == "__main__":
    raise SystemExit("请运行根目录 run.py --task full，以先生成本次主方案和基准结果。")
