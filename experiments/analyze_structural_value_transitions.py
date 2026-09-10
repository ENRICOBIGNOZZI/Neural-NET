"""Group-held-out forecasting of finite-horizon structural damage.

All predictors are fit to future *probe* damage only. Test damage is never a training label;
it is used solely to ask whether the probe-calibrated prediction transfers ex post.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _pearson(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def grouped_predictions(rows, features, *, alpha=1.0):
    groups = np.asarray([f"{r['dataset']}:{r['seed']}:{r['lr']}" for r in rows])
    y = np.asarray([r["future_probe_damage"] for r in rows], dtype=float)
    x = np.asarray([[r[f] for f in features] for r in rows], dtype=float)
    pred = np.full_like(y, np.nan)
    for group in np.unique(groups):
        train = groups != group
        test = groups == group
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        model.fit(x[train], y[train])
        pred[test] = model.predict(x[test])
    return y, pred


def metrics(y, pred):
    return {
        "r2": float(r2_score(y, pred)),
        "pearson": _pearson(y, pred),
        "mae": float(mean_absolute_error(y, pred)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("results/structural_value_transitions.json"))
    parser.add_argument("--output", type=Path, default=Path("results/structural_value_forecast.json"))
    args = parser.parse_args()

    payload = json.loads(args.input.read_text())
    rows = [
        r for r in payload["rows"]
        if np.isfinite(r["future_accessibility_gain_proxy"])
        and np.isfinite(r["target_angle_speed"])
    ]

    specs = {
        "time_and_size": ["step", "fraction_removed"],
        "current_damage_time_size": [
            "current_probe_damage",
            "step",
            "fraction_removed",
        ],
        "target_value_state": [
            "resolved_accessibility",
            "current_accessibility_loss",
            "target_angle_speed",
            "future_accessibility_gain_proxy",
            "saturation_ceiling",
            "effective_rank",
            "current_dense_refit_probe_mse",
            "current_probe_damage",
            "fraction_removed",
        ],
    }

    results = {}
    target_test = np.asarray([r["future_test_damage_ex_post"] for r in rows], dtype=float)
    for name, feats in specs.items():
        y, pred = grouped_predictions(rows, feats)
        results[name] = metrics(y, pred)
        results[name]["ex_post_test_damage_pearson"] = _pearson(target_test, pred)

    # Basic label sanity: how well does future probe structural damage itself track test damage?
    probe_damage = np.asarray([r["future_probe_damage"] for r in rows], dtype=float)
    results["label_audit"] = {
        "probe_vs_test_damage_pearson": _pearson(probe_damage, target_test),
        "n_transitions": len(rows),
        "groups": len(set(f"{r['dataset']}:{r['seed']}:{r['lr']}" for r in rows)),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"features": specs, "results": results}, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
