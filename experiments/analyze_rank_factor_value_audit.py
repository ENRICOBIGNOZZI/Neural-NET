"""Summarize the development rank-factor counterfactual audit."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _mean(xs):
    return float(np.mean(xs)) if xs else None


def _corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def main():
    files = sorted(Path("results").glob("rank_factor_audit_*_seed*.json"))
    if not files:
        raise SystemExit("no rank-factor audit files found")

    rows = []
    runs = []
    for path in files:
        payload = json.loads(path.read_text())
        runs.append(
            {
                "dataset": payload["dataset"],
                "seed": payload["seed"],
                "base_final_test_mse": payload["base_final_test_mse"],
                "base_final_test_acc": payload["base_final_test_acc"],
            }
        )
        rows.extend(payload["transitions"])

    methods = sorted({r["method"] for r in rows})
    method_summary = {}
    for method in methods:
        rr = [r for r in rows if r["method"] == method]
        method_summary[method] = {
            "n": len(rr),
            "mean_future_probe_damage": _mean([r["future_probe_damage"] for r in rr]),
            "mean_future_test_damage": _mean([r["future_test_damage"] for r in rr]),
            "median_future_test_damage": float(np.median([r["future_test_damage"] for r in rr])),
            "fraction_nonpositive_future_test_damage": _mean(
                [float(r["future_test_damage"] <= 0.0) for r in rr]
            ),
            "mean_current_test_damage": _mean([r["current_test_damage"] for r in rr]),
            "mean_dropped_group_gradient_fraction": _mean(
                [r["dropped_group_gradient_fraction"] for r in rr]
            ),
            "mean_dropped_scale_energy_fraction": _mean(
                [r["dropped_scale_energy_fraction"] for r in rr]
            ),
            "mean_parameter_saving_fraction": _mean(
                [r["parameter_saving_fraction"] for r in rr]
            ),
            "mean_continuation_speed_ratio": _mean(
                [r["continuation_speed_ratio"] for r in rr]
            ),
        }

    step_summary = {}
    for step in sorted({r["step"] for r in rows}):
        rr = [r for r in rows if r["step"] == step and r["method"] == "joint"]
        step_summary[str(step)] = {
            "n": len(rr),
            "joint_mean_future_test_damage": _mean([r["future_test_damage"] for r in rr]),
            "joint_mean_future_probe_damage": _mean([r["future_probe_damage"] for r in rr]),
            "mean_target_gradient_effective_rank": _mean(
                [r["target_gradient_effective_rank"] for r in rr]
            ),
            "mean_target_gradient_modes_95": _mean([r["target_gradient_modes_95"] for r in rr]),
            "mean_bottom_half_target_gradient_fraction": _mean(
                [r["bottom_half_target_gradient_fraction"] for r in rr]
            ),
        }

    nonrandom = [r for r in rows if r["method"] != "random"]
    diagnostics = {
        "corr_current_probe_vs_future_test_damage": _corr(
            [r["current_probe_damage"] for r in nonrandom],
            [r["future_test_damage"] for r in nonrandom],
        ),
        "corr_current_test_vs_future_test_damage": _corr(
            [r["current_test_damage"] for r in nonrandom],
            [r["future_test_damage"] for r in nonrandom],
        ),
        "corr_group_energy_fraction_vs_future_test_damage": _corr(
            [r["dropped_group_gradient_fraction"] for r in nonrandom],
            [r["future_test_damage"] for r in nonrandom],
        ),
        "corr_scale_energy_fraction_vs_future_test_damage": _corr(
            [r["dropped_scale_energy_fraction"] for r in nonrandom],
            [r["future_test_damage"] for r in nonrandom],
        ),
        "corr_structural_proxy_vs_future_test_damage": _corr(
            [r["structural_proxy_parameter_bound"] for r in nonrandom],
            [r["future_test_damage"] for r in nonrandom],
        ),
        "corr_target_gradient_effective_rank_vs_joint_damage": _corr(
            [r["target_gradient_effective_rank"] for r in rows if r["method"] == "joint"],
            [r["future_test_damage"] for r in rows if r["method"] == "joint"],
        ),
    }

    # Pair the three structured selection methods with random at the same run/checkpoint.
    paired_advantage = {}
    random_by_key = {
        (r["dataset"], r["seed"], r["step"]): r
        for r in rows
        if r["method"] == "random"
    }
    for method in ("joint", "gradient", "scale"):
        deltas = []
        for r in rows:
            if r["method"] != method:
                continue
            random_r = random_by_key[(r["dataset"], r["seed"], r["step"])]
            deltas.append(r["future_test_damage"] - random_r["future_test_damage"])
        paired_advantage[method] = {
            "mean_damage_minus_random": _mean(deltas),
            "fraction_beating_random": _mean([float(d <= 0.0) for d in deltas]),
        }

    summary = {
        "status": "development-only; no confirmatory claim",
        "n_runs": len(runs),
        "n_transitions": len(rows),
        "runs": runs,
        "method_summary": method_summary,
        "joint_onset_summary": step_summary,
        "diagnostics": diagnostics,
        "paired_selection_vs_random": paired_advantage,
    }
    out = Path("results/rank_factor_audit_summary.json")
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
