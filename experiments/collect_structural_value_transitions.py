"""Collect finite-horizon structural-damage counterfactuals on train/probe/test splits.

This is the empirical analogue of the master safe-contraction theorem.  At a dense
checkpoint t we create two branches from exactly the same state:

1. full-width ridge-refit + fresh AdamW;
2. structural contraction + the same ridge-refit + fresh AdamW.

Both branches train for H more updates.  Their *probe* risk difference is the development
outcome used to learn or calibrate a contraction rule.  Test risk is recorded only for
ex-post audit and must never be used to fit the forecaster.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from neural_net.feature_value import constant_speed_feature_value_proxy
from neural_net.models import ContractibleMLP, num_parameters
from neural_net.readout import fit_ridge_readout
from neural_net.spectral import (
    accessibility_from_features,
    effective_rank,
    target_angle_from_accessibility,
    target_greedy_neuron_order,
)
from run_dense_vs_sra import evaluate, features, load_task, step


def _branch_future(
    snapshot: ContractibleMLP,
    keep_idx: np.ndarray | None,
    xtr,
    ytr,
    xp,
    yp,
    xt,
    yt,
    *,
    lr: float,
    ridge: float,
    horizon: int,
):
    model = copy.deepcopy(snapshot)
    if keep_idx is not None:
        model = model.contracted_copy(keep_idx)
    fit_ridge_readout(model, xtr, ytr, alpha=ridge)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(horizon):
        step(model, opt, xtr, ytr)
    probe_mse, _ = evaluate(model, xp, yp)
    test_mse, _ = evaluate(model, xt, yt)
    return {
        "probe_mse": probe_mse,
        "test_mse": test_mse,
        "parameters": num_parameters(model),
    }


def collect_run(
    dataset: str,
    seed: int,
    lr: float,
    *,
    steps: int = 200,
    width: int = 64,
    checkpoints=(80, 120, 160),
    horizon: int = 40,
    fractions=(0.75, 0.875),
    resolved_rank: int = 5,
    ridge: float = 1e-3,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    xtr, ytr, xp, yp, xt, yt = load_task(dataset, seed)
    model = ContractibleMLP(xtr.shape[1], width, width)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    checkpoint_set = set(int(x) for x in checkpoints)
    previous_angle = None
    previous_step = None
    records = []

    for t in range(1, steps + 1):
        step(model, opt, xtr, ytr)
        if t not in checkpoint_set or t + horizon > steps:
            continue

        snapshot = copy.deepcopy(model)
        htr = features(snapshot, xtr)
        hp = features(snapshot, xp)
        q = accessibility_from_features(hp, yp.numpy(), rank=min(resolved_rank, width))
        angle = target_angle_from_accessibility(q)
        speed = float("nan")
        if previous_angle is not None:
            speed = abs(angle - previous_angle) / (t - previous_step)
        previous_angle, previous_step = angle, t
        future_proxy = (
            float("nan")
            if not np.isfinite(speed)
            else constant_speed_feature_value_proxy(q, speed, horizon)
        )
        reff = effective_rank(hp)
        current_network_probe_mse, _ = evaluate(snapshot, xp, yp)

        order = target_greedy_neuron_order(htr, ytr.numpy())
        # The horizon-zero dense branch is the fair current comparator: it receives
        # exactly the same ridge-readout action as every structural candidate.
        dense_current_refit = _branch_future(
            snapshot,
            None,
            xtr,
            ytr,
            xp,
            yp,
            xt,
            yt,
            lr=lr,
            ridge=ridge,
            horizon=0,
        )
        dense_future = _branch_future(
            snapshot,
            None,
            xtr,
            ytr,
            xp,
            yp,
            xt,
            yt,
            lr=lr,
            ridge=ridge,
            horizon=horizon,
        )

        for frac in fractions:
            rank = int(round(width * float(frac)))
            rank = max(1, min(rank, width - 1))
            keep_idx = order[:rank]
            q_candidate = accessibility_from_features(
                hp[:, keep_idx], yp.numpy(), rank=min(resolved_rank, rank)
            )
            candidate_current_refit = _branch_future(
                snapshot,
                keep_idx,
                xtr,
                ytr,
                xp,
                yp,
                xt,
                yt,
                lr=lr,
                ridge=ridge,
                horizon=0,
            )
            reduced_future = _branch_future(
                snapshot,
                keep_idx,
                xtr,
                ytr,
                xp,
                yp,
                xt,
                yt,
                lr=lr,
                ridge=ridge,
                horizon=horizon,
            )
            records.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "lr": lr,
                    "step": t,
                    "horizon": horizon,
                    "width": width,
                    "candidate_rank": rank,
                    "fraction_kept": rank / width,
                    "fraction_removed": 1.0 - rank / width,
                    "resolved_rank": min(resolved_rank, width),
                    "resolved_accessibility": q,
                    "candidate_accessibility": q_candidate,
                    "current_accessibility_loss": max(0.0, q - q_candidate),
                    "target_angle_speed": speed,
                    "future_accessibility_gain_proxy": future_proxy,
                    "saturation_ceiling": 1.0 - q,
                    "effective_rank": reff,
                    "current_network_probe_mse": current_network_probe_mse,
                    "current_dense_refit_probe_mse": dense_current_refit["probe_mse"],
                    "current_candidate_refit_probe_mse": candidate_current_refit["probe_mse"],
                    "current_probe_damage": candidate_current_refit["probe_mse"] - dense_current_refit["probe_mse"],
                    "future_dense_refit_probe_mse": dense_future["probe_mse"],
                    "future_reduced_probe_mse": reduced_future["probe_mse"],
                    "future_probe_damage": reduced_future["probe_mse"] - dense_future["probe_mse"],
                    "future_dense_refit_test_mse": dense_future["test_mse"],
                    "future_reduced_test_mse": reduced_future["test_mse"],
                    "future_test_damage_ex_post": reduced_future["test_mse"] - dense_future["test_mse"],
                    "dense_parameters": dense_future["parameters"],
                    "reduced_parameters": reduced_future["parameters"],
                    "parameter_reduction_fraction": 1.0 - reduced_future["parameters"] / dense_future["parameters"],
                }
            )
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["digits", "cancer"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--lrs", type=float, nargs="+", default=[0.01])
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--checkpoints", type=int, nargs="+", default=[80, 120, 160])
    parser.add_argument("--horizon", type=int, default=40)
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.75, 0.875])
    parser.add_argument("--output", type=Path, default=Path("results/structural_value_transitions.json"))
    args = parser.parse_args()

    rows = []
    for dataset in args.datasets:
        for lr in args.lrs:
            for seed in args.seeds:
                rows.extend(
                    collect_run(
                        dataset,
                        seed,
                        lr,
                        steps=args.steps,
                        width=args.width,
                        checkpoints=args.checkpoints,
                        horizon=args.horizon,
                        fractions=args.fractions,
                    )
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"config": vars(args) | {"output": str(args.output)}, "rows": rows}, indent=2))
    print(f"wrote {len(rows)} transitions to {args.output}")


if __name__ == "__main__":
    main()
