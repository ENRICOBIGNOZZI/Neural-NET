"""Small paired CPU pilot for the target-value controller."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from run_dense_vs_tv_sra import run


def _mean_present(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return None if not vals else float(np.mean(vals))


def main():
    configs = [("digits", 0.01), ("cancer", 0.01)]
    rows = []
    for dataset, lr in configs:
        for seed in range(4):
            rows.append(run(dataset, seed, lr, steps=200, width=64))

    summary = {}
    for dataset, _ in configs:
        subset = [r for r in rows if r["dataset"] == dataset]
        mean_refit = _mean_present(subset, "dense_refit_test_mse")
        mean_adaptive = float(np.mean([r["adaptive_test_mse"] for r in subset]))
        summary[dataset] = {
            "runs": len(subset),
            "contractions": int(sum(r["contracted"] for r in subset)),
            "mean_contraction_step": _mean_present(
                [r for r in subset if r["contracted"]], "contraction_step"
            ),
            "mean_test_mse_dense": float(np.mean([r["dense_test_mse"] for r in subset])),
            "mean_test_mse_adaptive": mean_adaptive,
            "mean_test_mse_optimizer_reset_control": _mean_present(subset, "dense_reset_test_mse"),
            "mean_test_mse_dense_readout_refit_control": mean_refit,
            "mean_adaptive_minus_dense_refit_mse": None if mean_refit is None else mean_adaptive - mean_refit,
            "mean_parameter_reduction_fraction": float(np.mean([r["parameter_reduction_fraction"] for r in subset])),
            "mean_param_step_saving_fraction": float(np.mean([r["param_step_saving_fraction"] for r in subset])),
            "mean_wallclock_ratio_including_controller": float(np.mean([r["adaptive_total_seconds"] / r["dense_train_seconds"] for r in subset])),
            "mean_controller_seconds": float(np.mean([r["controller_seconds"] for r in subset])),
        }

    out = {"summary": summary, "runs": rows}
    Path("results").mkdir(exist_ok=True)
    Path("results/tv_pilot_suite.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
