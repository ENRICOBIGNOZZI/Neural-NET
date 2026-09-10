from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from neural_net.bulk_spectrum import bulk_state_from_tangent_spectrum
from neural_net.rank_factor import RankContractibleMLP
from neural_net.torch_probe import empirical_tangent_spectrum


def half_mse(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    with torch.no_grad():
        residual = y - model(x)
        return 0.5 * float(torch.mean(residual.square()).cpu())


def train_one_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    y: torch.Tensor,
) -> None:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    residual = y - model(x)
    loss = 0.5 * torch.mean(residual.square())
    loss.backward()
    optimizer.step()


def train_full_batch(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    steps: int,
    lr: float,
) -> float:
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    start = time.perf_counter()
    for _ in range(int(steps)):
        train_one_step(model, optimizer, x, y)
    return float(time.perf_counter() - start)


def parameter_count(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def make_teacher_and_data(
    seed: int,
    *,
    input_dim: int,
    width1: int,
    width2: int,
    teacher_rank: int,
    train_n: int,
    spectral_n: int,
    selection_n: int,
    test_n: int,
    noise_std: float,
):
    generator = torch.Generator().manual_seed(10000 + int(seed))
    torch.manual_seed(20000 + int(seed))
    teacher = RankContractibleMLP(
        input_dim=input_dim,
        width1=width1,
        width2=width2,
        rank=teacher_rank,
    ).double()
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    total = train_n + spectral_n + selection_n + test_n
    x = torch.randn(total, input_dim, generator=generator, dtype=torch.float64)
    with torch.no_grad():
        y = teacher(x)
        y = (y - y.mean()) / y.std().clamp_min(1e-8)
        if noise_std > 0.0:
            y = y + noise_std * torch.randn(total, generator=generator, dtype=torch.float64)

    offsets = np.cumsum([0, train_n, spectral_n, selection_n, test_n])
    blocks = []
    for lo, hi in zip(offsets[:-1], offsets[1:]):
        blocks.append((x[int(lo) : int(hi)].clone(), y[int(lo) : int(hi)].clone()))
    return teacher, blocks


def oracle_keep_indices(
    model: RankContractibleMLP,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    keep_rank: int,
) -> tuple[tuple[int, ...], float]:
    """Best current structural subset on a held-out oracle selection sample."""
    if keep_rank < 1 or keep_rank >= model.rank:
        raise ValueError("keep_rank must be positive and smaller than model.rank")

    best_indices: tuple[int, ...] | None = None
    best_risk = float("inf")
    for indices in itertools.combinations(range(model.rank), keep_rank):
        candidate = model.contracted_rank_copy(indices)
        risk = half_mse(candidate, x, y)
        if risk < best_risk:
            best_risk = risk
            best_indices = tuple(int(j) for j in indices)
    if best_indices is None:
        raise RuntimeError("oracle subset search returned no candidate")
    return best_indices, float(best_risk)


def random_keep_indices(seed: int, checkpoint: int, rank: int, keep_rank: int) -> tuple[int, ...]:
    rng = np.random.default_rng(700000 + 1000 * int(seed) + int(checkpoint))
    keep = np.sort(rng.choice(rank, size=keep_rank, replace=False))
    return tuple(int(j) for j in keep)


def mean_ci95(values: list[float]) -> dict:
    x = np.asarray(values, dtype=float)
    n = int(x.size)
    mean = float(np.mean(x)) if n else float("nan")
    if n <= 1:
        return {"n": n, "mean": mean, "std": 0.0, "se": 0.0, "ci95_lo": mean, "ci95_hi": mean}
    std = float(np.std(x, ddof=1))
    se = std / math.sqrt(n)
    half = 1.96 * se
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "se": se,
        "ci95_lo": float(mean - half),
        "ci95_hi": float(mean + half),
    }


def run_seed(
    seed: int,
    *,
    input_dim: int,
    width1: int,
    width2: int,
    teacher_rank: int,
    student_rank: int,
    train_n: int,
    spectral_n: int,
    selection_n: int,
    test_n: int,
    noise_std: float,
    lr: float,
    horizon: int,
    checkpoints: list[int],
    damage_abs_tol: float,
    damage_rel_tol: float,
) -> dict:
    _, blocks = make_teacher_and_data(
        seed,
        input_dim=input_dim,
        width1=width1,
        width2=width2,
        teacher_rank=teacher_rank,
        train_n=train_n,
        spectral_n=spectral_n,
        selection_n=selection_n,
        test_n=test_n,
        noise_std=noise_std,
    )
    (train_x, train_y), (spectral_x, spectral_y), (select_x, select_y), (test_x, test_y) = blocks

    torch.manual_seed(30000 + int(seed))
    dense = RankContractibleMLP(
        input_dim=input_dim,
        width1=width1,
        width2=width2,
        rank=student_rank,
    ).double()
    initial_dense = copy.deepcopy(dense)

    # Strict compact baseline: give the compact model the same oracle selection
    # information at initialization that the later contraction receives.  Hence at
    # checkpoint 0, wide-then-contract and compact-from-start are the same model.
    initial_keep, initial_compact_selection_risk = oracle_keep_indices(
        initial_dense,
        select_x,
        select_y,
        keep_rank=teacher_rank,
    )
    compact = initial_dense.contracted_rank_copy(initial_keep)
    compact_optimizer = torch.optim.SGD(compact.parameters(), lr=lr)
    target_steps = {int(c + horizon) for c in checkpoints}
    max_target_step = max(target_steps)
    compact_risk_by_step: dict[int, float] = {0: half_mse(compact, test_x, test_y)}
    for step in range(1, max_target_step + 1):
        train_one_step(compact, compact_optimizer, train_x, train_y)
        if step in target_steps:
            compact_risk_by_step[step] = half_mse(compact, test_x, test_y)

    dense_optimizer = torch.optim.SGD(dense.parameters(), lr=lr)
    checkpoint_set = set(checkpoints)
    records: list[dict] = []

    for step in range(max(checkpoints) + 1):
        if step in checkpoint_set:
            checkpoint_model = copy.deepcopy(dense)
            current_dense_test = half_mse(checkpoint_model, test_x, test_y)
            current_dense_selection = half_mse(checkpoint_model, select_x, select_y)

            tangent = empirical_tangent_spectrum(checkpoint_model, spectral_x, spectral_y)
            bulk = bulk_state_from_tangent_spectrum(
                tangent,
                signal_rank=teacher_rank,
                horizon=float(horizon * lr),
            )

            oracle_keep, immediate_oracle_selection = oracle_keep_indices(
                checkpoint_model,
                select_x,
                select_y,
                keep_rank=teacher_rank,
            )
            random_keep = random_keep_indices(seed, step, student_rank, teacher_rank)

            dense_branch = copy.deepcopy(checkpoint_model)
            oracle_branch = checkpoint_model.contracted_rank_copy(oracle_keep)
            random_branch = checkpoint_model.contracted_rank_copy(random_keep)

            dense_wall = train_full_batch(dense_branch, train_x, train_y, steps=horizon, lr=lr)
            oracle_wall = train_full_batch(oracle_branch, train_x, train_y, steps=horizon, lr=lr)
            random_wall = train_full_batch(random_branch, train_x, train_y, steps=horizon, lr=lr)

            dense_future_test = half_mse(dense_branch, test_x, test_y)
            oracle_future_test = half_mse(oracle_branch, test_x, test_y)
            random_future_test = half_mse(random_branch, test_x, test_y)
            dense_future_selection = half_mse(dense_branch, select_x, select_y)
            oracle_future_selection = half_mse(oracle_branch, select_x, select_y)
            compact_future_test = compact_risk_by_step[int(step + horizon)]

            structural_damage = float(oracle_future_test - dense_future_test)
            damage_tolerance = float(max(damage_abs_tol, damage_rel_tol * dense_future_test))
            safe = bool(structural_damage <= damage_tolerance)
            width_value = float(compact_future_test - oracle_future_test)

            dense_params = parameter_count(dense_branch)
            reduced_params = parameter_count(oracle_branch)
            records.append(
                {
                    "seed": int(seed),
                    "checkpoint": int(step),
                    "horizon_steps": int(horizon),
                    "effective_horizon": float(horizon * lr),
                    "initial_oracle_keep_indices": list(initial_keep),
                    "initial_oracle_compact_selection_risk": float(initial_compact_selection_risk),
                    "current_dense_test_risk": current_dense_test,
                    "current_dense_selection_risk": current_dense_selection,
                    "oracle_keep_indices": list(oracle_keep),
                    "random_keep_indices": list(random_keep),
                    "immediate_oracle_contraction_selection_damage": float(
                        immediate_oracle_selection - current_dense_selection
                    ),
                    "future_dense_test_risk": dense_future_test,
                    "future_oracle_reduced_test_risk": oracle_future_test,
                    "future_random_reduced_test_risk": random_future_test,
                    "future_test_structural_damage": structural_damage,
                    "future_dense_selection_risk": dense_future_selection,
                    "future_oracle_reduced_selection_risk": oracle_future_selection,
                    "future_selection_structural_damage": float(
                        oracle_future_selection - dense_future_selection
                    ),
                    "damage_tolerance": damage_tolerance,
                    "safe_contraction": safe,
                    "compact_from_start_test_risk_at_same_step": compact_future_test,
                    "wide_then_contract_minus_compact_from_start": float(
                        oracle_future_test - compact_future_test
                    ),
                    "width_path_value": width_value,
                    "oracle_gain_vs_random_contraction": float(random_future_test - oracle_future_test),
                    "signal_floor": bulk.signal_floor,
                    "bulk_edge": bulk.bulk_edge,
                    "eigengap": bulk.eigengap,
                    "bulk_target_mass": bulk.bulk_target_mass,
                    "bulk_target_mass_fraction": bulk.bulk_target_mass_fraction,
                    "bulk_gradient_energy": bulk.bulk_gradient_energy,
                    "bulk_gradient_energy_fraction": bulk.bulk_gradient_energy_fraction,
                    "bulk_fixed_geometry_value": bulk.bulk_fixed_geometry_value,
                    "dense_parameters": dense_params,
                    "reduced_parameters": reduced_params,
                    "parameter_fraction_removed": float(1.0 - reduced_params / dense_params),
                    "dense_horizon_parameter_steps": int(dense_params * horizon),
                    "reduced_horizon_parameter_steps": int(reduced_params * horizon),
                    "parameter_step_fraction_saved": float(1.0 - reduced_params / dense_params),
                    "dense_continuation_wall_seconds": dense_wall,
                    "oracle_reduced_continuation_wall_seconds": oracle_wall,
                    "random_reduced_continuation_wall_seconds": random_wall,
                }
            )

        if step < max(checkpoints):
            train_one_step(dense, dense_optimizer, train_x, train_y)

    persistent_onset: int | None = None
    for i, record in enumerate(records):
        if record["safe_contraction"] and all(r["safe_contraction"] for r in records[i:]):
            persistent_onset = int(record["checkpoint"])
            break

    return {
        "seed": int(seed),
        "initial_dense_parameters": parameter_count(initial_dense),
        "initial_compact_parameters": parameter_count(compact),
        "initial_oracle_keep_indices": list(initial_keep),
        "persistent_safe_onset_checkpoint": persistent_onset,
        "records": records,
    }


def summarize(results: list[dict]) -> dict:
    rows = [row for result in results for row in result["records"]]
    checkpoints = sorted({int(row["checkpoint"]) for row in rows})
    by_checkpoint: dict[str, dict] = {}

    for checkpoint in checkpoints:
        group = [row for row in rows if int(row["checkpoint"]) == checkpoint]
        by_checkpoint[str(checkpoint)] = {
            "future_test_structural_damage": mean_ci95(
                [float(row["future_test_structural_damage"]) for row in group]
            ),
            "wide_then_contract_minus_compact_from_start": mean_ci95(
                [float(row["wide_then_contract_minus_compact_from_start"]) for row in group]
            ),
            "width_path_value": mean_ci95([float(row["width_path_value"]) for row in group]),
            "oracle_gain_vs_random_contraction": mean_ci95(
                [float(row["oracle_gain_vs_random_contraction"]) for row in group]
            ),
            "bulk_target_mass": mean_ci95([float(row["bulk_target_mass"]) for row in group]),
            "bulk_gradient_energy": mean_ci95([float(row["bulk_gradient_energy"]) for row in group]),
            "bulk_fixed_geometry_value": mean_ci95(
                [float(row["bulk_fixed_geometry_value"]) for row in group]
            ),
            "eigengap": mean_ci95([float(row["eigengap"]) for row in group]),
            "safe_fraction": float(np.mean([bool(row["safe_contraction"]) for row in group])),
            "width_value_positive_fraction": float(
                np.mean([float(row["width_path_value"]) > 0.0 for row in group])
            ),
            "parameter_step_fraction_saved": float(
                np.mean([float(row["parameter_step_fraction_saved"]) for row in group])
            ),
        }

    damage = np.asarray([float(row["future_test_structural_damage"]) for row in rows], dtype=float)
    correlations: dict[str, float] = {}
    for key in [
        "bulk_target_mass",
        "bulk_target_mass_fraction",
        "bulk_gradient_energy",
        "bulk_gradient_energy_fraction",
        "bulk_fixed_geometry_value",
        "eigengap",
    ]:
        x = np.asarray([float(row[key]) for row in rows], dtype=float)
        correlations[key] = (
            float(np.corrcoef(x, damage)[0, 1])
            if x.size >= 3 and np.std(x) > 0.0 and np.std(damage) > 0.0
            else 0.0
        )

    onsets = [
        result["persistent_safe_onset_checkpoint"]
        for result in results
        if result["persistent_safe_onset_checkpoint"] is not None
    ]
    return {
        "by_checkpoint": by_checkpoint,
        "pearson_with_future_test_structural_damage": correlations,
        "persistent_safe_onset": {
            "resolved_fraction": float(len(onsets) / len(results)) if results else 0.0,
            "onsets": onsets,
            "mean": float(np.mean(onsets)) if onsets else None,
            "median": float(np.median(onsets)) if onsets else None,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--input-dim", type=int, default=5)
    parser.add_argument("--width1", type=int, default=12)
    parser.add_argument("--width2", type=int, default=12)
    parser.add_argument("--teacher-rank", type=int, default=2)
    parser.add_argument("--student-rank", type=int, default=8)
    parser.add_argument("--train-n", type=int, default=256)
    parser.add_argument("--spectral-n", type=int, default=48)
    parser.add_argument("--selection-n", type=int, default=512)
    parser.add_argument("--test-n", type=int, default=2048)
    parser.add_argument("--noise-std", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--horizon", type=int, default=40)
    parser.add_argument("--damage-abs-tol", type=float, default=1e-3)
    parser.add_argument("--damage-rel-tol", type=float, default=0.05)
    parser.add_argument("--checkpoints", type=int, nargs="+", default=[0, 20, 40, 60, 80, 120])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/strict_oracle_onset_sweep.json"),
    )
    args = parser.parse_args()

    if args.teacher_rank >= args.student_rank:
        raise ValueError("teacher-rank must be smaller than student-rank")
    if sorted(set(args.checkpoints)) != args.checkpoints or args.checkpoints[0] < 0:
        raise ValueError("checkpoints must be strictly increasing nonnegative integers")
    if args.damage_abs_tol < 0.0 or args.damage_rel_tol < 0.0:
        raise ValueError("damage tolerances must be nonnegative")

    runs = [
        run_seed(
            seed,
            input_dim=args.input_dim,
            width1=args.width1,
            width2=args.width2,
            teacher_rank=args.teacher_rank,
            student_rank=args.student_rank,
            train_n=args.train_n,
            spectral_n=args.spectral_n,
            selection_n=args.selection_n,
            test_n=args.test_n,
            noise_std=args.noise_std,
            lr=args.lr,
            horizon=args.horizon,
            checkpoints=args.checkpoints,
            damage_abs_tol=args.damage_abs_tol,
            damage_rel_tol=args.damage_rel_tol,
        )
        for seed in args.seeds
    ]

    payload = {
        "status": "strict_development_oracle_onset_diagnostic",
        "warning": (
            "Both the initial compact control and checkpoint contraction use exhaustive target-aware "
            "subset search on an independent selection sample. The tangent bulk split also uses the "
            "known teacher rank. This is an oracle scientific diagnostic, not a deployable controller."
        ),
        "config": vars(args) | {"output": str(args.output)},
        "runs": runs,
        "summary": summarize(runs),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
