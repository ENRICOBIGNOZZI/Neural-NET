"""Dense AdamW versus second-generation target-value structural contraction.

This pilot changes only the *onset statistic* and maximum one-shot contraction size relative
to the first SRA audit. It keeps the same data split, optimizer, physical last-hidden-layer
contraction, ridge readout action, and full-width controls.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from neural_net.models import ContractibleMLP, num_parameters
from neural_net.readout import fit_ridge_readout
from neural_net.target_value_controller import TargetValueConfig, TargetValueContractionController
from run_dense_vs_sra import evaluate, features, load_task, step


def run(dataset: str, seed: int, lr: float, steps: int, width: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    xtr, ytr, xp, yp, xt, yt = load_task(dataset, seed)

    initial = ContractibleMLP(xtr.shape[1], width, width)
    dense = copy.deepcopy(initial)
    adaptive = copy.deepcopy(initial)
    dense_opt = torch.optim.AdamW(dense.parameters(), lr=lr, weight_decay=1e-4)
    adaptive_opt = torch.optim.AdamW(adaptive.parameters(), lr=lr, weight_decay=1e-4)

    cfg = TargetValueConfig(
        min_step=max(60, steps // 4),
        checkpoint_interval=20,
        resolved_rank=min(5, width),
        future_horizon_steps=40,
        max_future_accessibility_gain=0.005,
        min_rank=max(4, width // 8),
        rank_stride=4,
        max_accessibility_loss=0.01,
        max_probe_risk_increase=0.01,
        max_fraction_removed=0.25,
        persistence=2,
        ridge=1e-3,
    )
    controller = TargetValueContractionController(cfg)

    contracted = False
    contraction_step = None
    contraction_decision = None
    dense_reset = None
    dense_reset_opt = None
    dense_refit = None
    dense_refit_opt = None

    dense_param_steps = adaptive_param_steps = 0
    dense_reset_param_steps = dense_refit_param_steps = 0
    controller_seconds = action_seconds = 0.0
    dense_seconds = adaptive_seconds = 0.0
    dense_reset_seconds = dense_refit_seconds = 0.0
    checkpoints = []

    for t in range(1, steps + 1):
        tic = time.perf_counter()
        step(dense, dense_opt, xtr, ytr)
        dense_seconds += time.perf_counter() - tic
        dense_param_steps += num_parameters(dense)

        tic = time.perf_counter()
        step(adaptive, adaptive_opt, xtr, ytr)
        adaptive_seconds += time.perf_counter() - tic
        adaptive_param_steps += num_parameters(adaptive)

        if dense_reset is not None:
            tic = time.perf_counter()
            step(dense_reset, dense_reset_opt, xtr, ytr)
            dense_reset_seconds += time.perf_counter() - tic
            dense_reset_param_steps += num_parameters(dense_reset)

        if dense_refit is not None:
            tic = time.perf_counter()
            step(dense_refit, dense_refit_opt, xtr, ytr)
            dense_refit_seconds += time.perf_counter() - tic
            dense_refit_param_steps += num_parameters(dense_refit)

        if (not contracted) and t % cfg.checkpoint_interval == 0:
            tic = time.perf_counter()
            htr = features(adaptive, xtr)
            hp = features(adaptive, xp)
            decision = controller.evaluate(t, htr, ytr.numpy(), hp, yp.numpy())
            controller_seconds += time.perf_counter() - tic
            checkpoints.append(asdict(decision) | {"keep_indices": None})

            if decision.should_contract:
                contraction_step = t
                contraction_decision = decision

                dense_reset = copy.deepcopy(adaptive)
                dense_refit = copy.deepcopy(adaptive)
                dense_reset_opt = torch.optim.AdamW(
                    dense_reset.parameters(), lr=lr, weight_decay=1e-4
                )

                tic = time.perf_counter()
                fit_ridge_readout(dense_refit, xtr, ytr, alpha=cfg.ridge)
                dense_refit_opt = torch.optim.AdamW(
                    dense_refit.parameters(), lr=lr, weight_decay=1e-4
                )

                adaptive = adaptive.contracted_copy(decision.keep_indices)
                fit_ridge_readout(adaptive, xtr, ytr, alpha=cfg.ridge)
                adaptive_opt = torch.optim.AdamW(
                    adaptive.parameters(), lr=lr, weight_decay=1e-4
                )
                action_seconds += time.perf_counter() - tic
                contracted = True

    dense_test = evaluate(dense, xt, yt)
    adaptive_test = evaluate(adaptive, xt, yt)
    reset_test = evaluate(dense_reset, xt, yt) if dense_reset is not None else None
    refit_test = evaluate(dense_refit, xt, yt) if dense_refit is not None else None

    return {
        "dataset": dataset,
        "seed": seed,
        "lr": lr,
        "steps": steps,
        "initial_width": width,
        "contracted": contracted,
        "contraction_step": contraction_step,
        "final_width": adaptive.width2,
        "dense_parameters": num_parameters(dense),
        "adaptive_parameters": num_parameters(adaptive),
        "parameter_reduction_fraction": 1 - num_parameters(adaptive) / num_parameters(dense),
        "dense_test_mse": dense_test[0],
        "adaptive_test_mse": adaptive_test[0],
        "dense_refit_test_mse": None if refit_test is None else refit_test[0],
        "dense_reset_test_mse": None if reset_test is None else reset_test[0],
        "dense_param_steps": dense_param_steps,
        "adaptive_param_steps": adaptive_param_steps,
        "param_step_saving_fraction": 1 - adaptive_param_steps / dense_param_steps,
        "dense_train_seconds": dense_seconds,
        "adaptive_train_seconds": adaptive_seconds,
        "controller_seconds": controller_seconds,
        "action_seconds": action_seconds,
        "adaptive_total_seconds": adaptive_seconds + controller_seconds + action_seconds,
        "contraction_decision": None if contraction_decision is None else {
            k: v for k, v in asdict(contraction_decision).items() if k != "keep_indices"
        },
        "controller_config": asdict(cfg),
        "checkpoints": checkpoints,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["digits", "cancer"], default="digits")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--width", type=int, default=64)
    args = parser.parse_args()
    result = run(args.dataset, args.seed, args.lr, args.steps, args.width)
    Path("results").mkdir(exist_ok=True)
    path = Path(f"results/tv_sra_{args.dataset}_seed{args.seed}.json")
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "checkpoints"}, indent=2))


if __name__ == "__main__":
    main()
