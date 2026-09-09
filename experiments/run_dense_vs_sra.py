"""Leakage-free pilot: dense AdamW versus theory-derived structural contraction.

The controller sees train/probe information only. Test labels are used ex post.
The first implementation contracts only the final hidden layer, where removing neurons is
structurally exact and produces a real parameter/FLOP reduction. It is intentionally a pilot,
not a claim of universal acceleration.
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
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from neural_net.controller import ContractionConfig, FeatureSpanContractionController
from neural_net.models import ContractibleMLP, num_parameters


def load_task(name: str, seed: int):
    if name == "digits":
        data = load_digits()
        x = data.data.astype(np.float32)
        y = np.where(data.target % 2 == 0, 1.0, -1.0).astype(np.float32)
    elif name == "cancer":
        data = load_breast_cancer()
        x = data.data.astype(np.float32)
        y = np.where(data.target == 1, 1.0, -1.0).astype(np.float32)
    else:
        raise ValueError(name)

    x_train, x_tmp, y_train, y_tmp = train_test_split(
        x, y, test_size=0.40, random_state=seed, stratify=y
    )
    x_probe, x_test, y_probe, y_test = train_test_split(
        x_tmp, y_tmp, test_size=0.50, random_state=seed + 1, stratify=y_tmp
    )
    scaler = StandardScaler().fit(x_train)
    return tuple(
        torch.tensor(arr, dtype=torch.float32)
        for arr in (
            scaler.transform(x_train), y_train,
            scaler.transform(x_probe), y_probe,
            scaler.transform(x_test), y_test,
        )
    )


def step(model, opt, x, y):
    model.train()
    opt.zero_grad(set_to_none=True)
    pred = model(x)
    loss = torch.mean((pred - y) ** 2)
    loss.backward()
    opt.step()
    return float(loss.detach())


def evaluate(model, x, y):
    model.eval()
    with torch.no_grad():
        pred = model(x)
        mse = float(torch.mean((pred - y) ** 2))
        acc = float(((pred >= 0) == (y >= 0)).float().mean())
    return mse, acc


def features(model, x):
    model.eval()
    with torch.no_grad():
        return model.features(x).cpu().numpy()


def run(dataset: str, seed: int, lr: float, steps: int, width: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    xtr, ytr, xp, yp, xt, yt = load_task(dataset, seed)

    initial = ContractibleMLP(xtr.shape[1], width, width)
    dense = copy.deepcopy(initial)
    adaptive = copy.deepcopy(initial)
    dense_opt = torch.optim.AdamW(dense.parameters(), lr=lr, weight_decay=1e-4)
    adaptive_opt = torch.optim.AdamW(adaptive.parameters(), lr=lr, weight_decay=1e-4)

    cfg = ContractionConfig(
        min_step=max(40, steps // 4),
        checkpoint_interval=20,
        min_rank=max(4, width // 8),
        max_accessibility_loss=0.01,
        max_probe_risk_increase=0.01,
        max_subspace_speed=0.006,
        speed_rank=min(8, width),
        rank_stride=4,
        persistence=2,
        ridge=1e-3,
    )
    controller = FeatureSpanContractionController(cfg)

    contracted = False
    contraction_step = None
    contraction_decision = None
    dense_reset = None
    dense_reset_opt = None

    dense_param_steps = 0
    adaptive_param_steps = 0
    dense_reset_param_steps = 0
    controller_seconds = 0.0
    dense_seconds = 0.0
    adaptive_seconds = 0.0

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
            step(dense_reset, dense_reset_opt, xtr, ytr)
            dense_reset_param_steps += num_parameters(dense_reset)

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

                # Optimizer-reset control: same checkpoint, same width, fresh AdamW state.
                dense_reset = copy.deepcopy(adaptive)
                dense_reset_opt = torch.optim.AdamW(
                    dense_reset.parameters(), lr=lr, weight_decay=1e-4
                )

                adaptive = adaptive.contracted_copy(decision.keep_indices)
                adaptive_opt = torch.optim.AdamW(
                    adaptive.parameters(), lr=lr, weight_decay=1e-4
                )
                contracted = True

    dense_test = evaluate(dense, xt, yt)
    adaptive_test = evaluate(adaptive, xt, yt)
    dense_probe = evaluate(dense, xp, yp)
    adaptive_probe = evaluate(adaptive, xp, yp)
    reset_test = evaluate(dense_reset, xt, yt) if dense_reset is not None else None

    result = {
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
        "dense_test_accuracy": dense_test[1],
        "adaptive_test_accuracy": adaptive_test[1],
        "dense_probe_mse": dense_probe[0],
        "adaptive_probe_mse": adaptive_probe[0],
        "dense_reset_test_mse": None if reset_test is None else reset_test[0],
        "dense_param_steps": dense_param_steps,
        "adaptive_param_steps": adaptive_param_steps,
        "param_step_saving_fraction": 1 - adaptive_param_steps / dense_param_steps,
        "dense_train_seconds": dense_seconds,
        "adaptive_train_seconds": adaptive_seconds,
        "controller_seconds": controller_seconds,
        "adaptive_total_seconds": adaptive_seconds + controller_seconds,
        "contraction_decision": None if contraction_decision is None else {
            "proposed_rank": contraction_decision.proposed_rank,
            "full_accessibility": contraction_decision.full_accessibility,
            "proposed_accessibility": contraction_decision.proposed_accessibility,
            "full_probe_mse": contraction_decision.full_probe_mse,
            "proposed_probe_mse": contraction_decision.proposed_probe_mse,
            "subspace_speed": contraction_decision.subspace_speed,
            "effective_rank": contraction_decision.effective_rank,
        },
        "controller_config": asdict(cfg),
        "checkpoints": checkpoints,
    }
    return result


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
    path = Path(f"results/{args.dataset}_seed{args.seed}.json")
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "checkpoints"}, indent=2))


if __name__ == "__main__":
    main()
