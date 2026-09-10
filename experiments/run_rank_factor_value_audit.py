"""Counterfactual audit for physically annealable rank-factor layers.

This is a development experiment, not a certified controller.  At several checkpoints we:

1. copy and canonically gauge the current full-rank model;
2. measure exact structural group-gradient energy on training data;
3. construct rank-drop candidates without using probe/test labels;
4. physically remove rank-one factors;
5. continue the compact and matched dense snapshots for the same horizon with fresh SGD;
6. measure future paired structural damage on probe and test data.

The experiment asks whether the quantities entering the new structural theorem actually become
small when later rank contraction becomes safe.  It also records a small empirical tangent
spectrum so that structural compressibility can be compared with target-weighted prediction
compressibility.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from neural_net.models import num_parameters
from neural_net.rank_factor import RankContractibleMLP, rank_factor_group_gradient_energy
from neural_net.torch_probe import empirical_tangent_spectrum
from run_dense_vs_sra import evaluate, load_task


def half_mse_step(model, opt, x, y) -> float:
    model.train()
    opt.zero_grad(set_to_none=True)
    pred = model(x)
    loss = 0.5 * torch.mean((pred - y) ** 2)
    loss.backward()
    opt.step()
    return float(loss.detach())


def train_horizon(model, x, y, *, lr: float, steps: int) -> float:
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    tic = time.perf_counter()
    for _ in range(steps):
        half_mse_step(model, opt, x, y)
    return float(time.perf_counter() - tic)


def checkpoint_group_state(model, x, y):
    model.zero_grad(set_to_none=True)
    pred = model(x)
    risk = 0.5 * torch.mean((pred - y) ** 2)
    risk.backward()
    group = rank_factor_group_gradient_energy(model.hidden2).detach().cpu().numpy()
    grad_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            grad_sq += float(p.grad.detach().square().sum())
    scale = model.hidden2.component_frobenius_norms().detach().cpu().numpy()
    return {
        "risk": float(risk.detach()),
        "gradient_norm": math.sqrt(max(grad_sq, 0.0)),
        "group_gradient_energy": group,
        "component_frobenius_norm": scale,
    }


def target_weighted_spectral_state(model, x, y, sample_size: int):
    n = min(int(sample_size), int(x.shape[0]))
    spec = empirical_tangent_spectrum(model, x[:n], y[:n])
    mu = np.asarray(spec.eigenvalues, dtype=float)
    a = np.asarray(spec.coefficients, dtype=float)
    w = np.maximum(mu * a * a, 0.0)
    total_w = float(w.sum())
    positive = mu > max(1e-12, 1e-10 * float(mu.max() if len(mu) else 0.0))

    if total_w > 0.0:
        p = w[w > 0.0] / total_w
        target_erank = float(np.exp(-(p * np.log(p)).sum()))
        ranked_w = np.sort(w)[::-1]
        top95 = int(np.searchsorted(np.cumsum(ranked_w), 0.95 * total_w) + 1)
    else:
        target_erank = 0.0
        top95 = 0

    pos_mu = mu[positive]
    if pos_mu.size:
        p_mu = pos_mu / pos_mu.sum()
        spectral_erank = float(np.exp(-(p_mu * np.log(p_mu)).sum()))
    else:
        spectral_erank = 0.0

    bottom_half_fraction = 0.0
    if pos_mu.size and total_w > 0.0:
        pos_indices = np.flatnonzero(positive)
        bottom = pos_indices[len(pos_indices) // 2 :]
        bottom_half_fraction = float(w[bottom].sum() / total_w)

    max_gap_ratio = 1.0
    max_gap_index = 0
    if pos_mu.size >= 2:
        ratios = pos_mu[:-1] / np.maximum(pos_mu[1:], 1e-30)
        max_gap_index = int(np.argmax(ratios) + 1)
        max_gap_ratio = float(np.max(ratios))

    return {
        "spectral_sample_size": n,
        "residual_risk": float(spec.residual_risk),
        "positive_tangent_rank": int(pos_mu.size),
        "tangent_effective_rank": spectral_erank,
        "target_gradient_effective_rank": target_erank,
        "target_gradient_modes_95": top95,
        "bottom_half_target_gradient_fraction": bottom_half_fraction,
        "largest_adjacent_eigenvalue_ratio": max_gap_ratio,
        "largest_gap_kept_rank": max_gap_index,
        "top_eigenvalues": mu[: min(10, len(mu))].tolist(),
        "top_target_gradient_weights": w[: min(10, len(w))].tolist(),
    }


def choose_drop_set(method: str, state, *, drop_count: int, rng: np.random.Generator):
    scale = np.asarray(state["component_frobenius_norm"], dtype=float)
    energy = np.asarray(state["group_gradient_energy"], dtype=float)
    if method == "scale":
        score = scale * scale
    elif method == "gradient":
        score = energy
    elif method == "joint":
        score = scale * scale / max(float(np.sum(scale * scale)), 1e-30)
        score = score + energy / max(float(np.sum(energy)), 1e-30)
    elif method == "random":
        return np.sort(rng.choice(len(scale), size=drop_count, replace=False))
    else:
        raise ValueError(method)
    return np.sort(np.argsort(score)[:drop_count])


def run(
    dataset: str,
    seed: int,
    *,
    lr: float,
    steps: int,
    width: int,
    rank: int,
    horizon_steps: int,
    drop_fraction: float,
    checkpoints: list[int],
    spectral_sample_size: int,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)
    xtr, ytr, xp, yp, xt, yt = load_task(dataset, seed)

    model = RankContractibleMLP(xtr.shape[1], width, width, rank)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    drop_count = min(rank - 1, max(1, int(round(rank * drop_fraction))))
    methods = ("joint", "gradient", "scale", "random")
    checkpoint_set = set(checkpoints)
    transitions = []
    base_seconds = 0.0

    for t in range(1, steps + 1):
        tic = time.perf_counter()
        half_mse_step(model, opt, xtr, ytr)
        base_seconds += time.perf_counter() - tic

        if t not in checkpoint_set:
            continue

        # The audit fixes the rank-factor gauge only on a copied checkpoint.  Both
        # dense and compact counterfactuals then receive a fresh optimizer so the
        # comparison is not contaminated by an untransformed SGD/Adam state.
        snapshot = copy.deepcopy(model).canonicalize_rank_components_()
        state = checkpoint_group_state(snapshot, xtr, ytr)
        spectral = target_weighted_spectral_state(snapshot, xp, yp, spectral_sample_size)

        dense_future = copy.deepcopy(snapshot)
        dense_seconds = train_horizon(dense_future, xtr, ytr, lr=lr, steps=horizon_steps)
        dense_probe_future = evaluate(dense_future, xp, yp)
        dense_test_future = evaluate(dense_future, xt, yt)
        dense_probe_now = evaluate(snapshot, xp, yp)
        dense_test_now = evaluate(snapshot, xt, yt)

        h_flow = lr * horizon_steps
        scale = np.asarray(state["component_frobenius_norm"], dtype=float)
        energy = np.asarray(state["group_gradient_energy"], dtype=float)
        total_energy = max(float(energy.sum()), 1e-30)
        total_scale_sq = max(float(np.sum(scale * scale)), 1e-30)

        for method_id, method in enumerate(methods):
            rng = np.random.default_rng(seed * 1000003 + t * 101 + method_id)
            drop = choose_drop_set(method, state, drop_count=drop_count, rng=rng)
            keep = np.asarray([q for q in range(rank) if q not in set(drop.tolist())], dtype=int)
            reduced = snapshot.contracted_rank_copy(keep)

            reduced_probe_now = evaluate(reduced, xp, yp)
            reduced_test_now = evaluate(reduced, xt, yt)
            reduced_seconds = train_horizon(reduced, xtr, ytr, lr=lr, steps=horizon_steps)
            reduced_probe_future = evaluate(reduced, xp, yp)
            reduced_test_future = evaluate(reduced, xt, yt)

            d0 = float(np.linalg.norm(scale[drop]))
            e_drop = float(energy[drop].sum())
            proxy = d0 + h_flow * math.sqrt(max(e_drop, 0.0))
            saved_fraction = 1.0 - num_parameters(reduced) / num_parameters(snapshot)

            transitions.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "step": t,
                    "method": method,
                    "initial_rank": rank,
                    "kept_rank": int(len(keep)),
                    "drop_count": int(len(drop)),
                    "drop_indices": drop.tolist(),
                    "flow_horizon": h_flow,
                    "checkpoint_train_risk": state["risk"],
                    "checkpoint_gradient_norm": state["gradient_norm"],
                    "dropped_scale_norm": d0,
                    "dropped_group_gradient_energy": e_drop,
                    "dropped_group_gradient_fraction": e_drop / total_energy,
                    "dropped_scale_energy_fraction": float(np.sum(scale[drop] ** 2)) / total_scale_sq,
                    "structural_proxy_parameter_bound": proxy,
                    "parameter_saving_fraction": saved_fraction,
                    "current_probe_damage": reduced_probe_now[0] - dense_probe_now[0],
                    "current_test_damage": reduced_test_now[0] - dense_test_now[0],
                    "future_probe_damage": reduced_probe_future[0] - dense_probe_future[0],
                    "future_test_damage": reduced_test_future[0] - dense_test_future[0],
                    "future_probe_acc_change": reduced_probe_future[1] - dense_probe_future[1],
                    "future_test_acc_change": reduced_test_future[1] - dense_test_future[1],
                    "dense_future_probe_mse": dense_probe_future[0],
                    "reduced_future_probe_mse": reduced_probe_future[0],
                    "dense_future_test_mse": dense_test_future[0],
                    "reduced_future_test_mse": reduced_test_future[0],
                    "dense_future_seconds": dense_seconds,
                    "reduced_future_seconds": reduced_seconds,
                    "continuation_speed_ratio": reduced_seconds / max(dense_seconds, 1e-12),
                }
                | spectral
            )

    final_test = evaluate(model, xt, yt)
    return {
        "dataset": dataset,
        "seed": seed,
        "lr": lr,
        "steps": steps,
        "width": width,
        "rank": rank,
        "horizon_steps": horizon_steps,
        "drop_fraction": drop_fraction,
        "checkpoints": checkpoints,
        "spectral_sample_size": spectral_sample_size,
        "base_train_seconds": base_seconds,
        "base_final_test_mse": final_test[0],
        "base_final_test_acc": final_test[1],
        "transitions": transitions,
        "status": "development-only counterfactual audit",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["digits", "cancer"], default="digits")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--horizon-steps", type=int, default=40)
    parser.add_argument("--drop-fraction", type=float, default=0.25)
    parser.add_argument("--checkpoints", type=int, nargs="+", default=[60, 120, 180])
    parser.add_argument("--spectral-sample-size", type=int, default=32)
    args = parser.parse_args()

    result = run(
        args.dataset,
        args.seed,
        lr=args.lr,
        steps=args.steps,
        width=args.width,
        rank=args.rank,
        horizon_steps=args.horizon_steps,
        drop_fraction=args.drop_fraction,
        checkpoints=args.checkpoints,
        spectral_sample_size=args.spectral_sample_size,
    )
    Path("results").mkdir(exist_ok=True)
    path = Path(f"results/rank_factor_audit_{args.dataset}_seed{args.seed}.json")
    path.write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            {
                "dataset": args.dataset,
                "seed": args.seed,
                "base_final_test_mse": result["base_final_test_mse"],
                "base_final_test_acc": result["base_final_test_acc"],
                "n_transitions": len(result["transitions"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
