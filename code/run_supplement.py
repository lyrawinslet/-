"""两项补充实验的运行与独立核验；调用与原实验相同的模型函数。"""
from __future__ import annotations
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def run(task: str, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    jobs = [("forecast", "run_validation.py", "verify_validation.py", "results"),
            ("storage", "run_experiment.py", "verify_experiment.py", "full_run")]
    for name, runner, verifier, saved in jobs:
        folder = ROOT / "experiments" / name
        if task == "supplement":
            results = output / name
            subprocess.run([sys.executable, str(folder / runner),
                            "--output", str(results)], cwd=ROOT, check=True)
        elif task == "supplement_verify":
            results = folder / saved
        else:
            raise ValueError(f"Unknown supplementary task: {task}")
        report = output / f"{name}_verification.json"
        subprocess.run([sys.executable, str(folder / verifier),
                        "--output", str(results), "--report", str(report)],
                       cwd=ROOT, check=True)
    print("两项补充实验核验完成，输出目录：", output, flush=True)
