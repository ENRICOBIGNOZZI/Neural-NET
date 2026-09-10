from __future__ import annotations

import argparse
import copy
import itertools
import json
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
    model.train()
    for _ in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        residual = y - model(x)
        loss = 0.5 * torch.mean(residual.square())
        loss.backward()
        optimizer.step()
    return float(time.perf_counter() - start)


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
    """Best current structural subset on an independent oracle selection sample.

    This deliberately uses target labels and exhaustive subset search.  It is not a
    deployable controller.  Its purpose is to isolate the scientific question of
    *when* contraction becomes safe from the separate question of how to discover
    the best structural subset cheaply.
    """
    if keep_rank < 1 or keep_rank >= model.rank:
        raise ValueError("keep_rank must be positive and smaller than the current rank")

    best_indices: tuple[int, ...] | None = None
    best_risk = float("inf")
    for indices in itertools.combinations(range(model.rank), keep_rank):
        candidate = model.contracted_rank_copy(indices)
        risk = half_mse(candidate, x, y)
        if risk < best_risk:
            best_risk = risk
            best_indices = tuple(int(j) for j in indices)
    assert best_indices is not None
    return best_indices, float(best_risk)


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

    # Compact-from-start control shares hidden1/out and an actual subset of the same
    # initial overparameterized model.  No future checkpoint information is used.
    compact = initial_dense.contracted_rank_copy(tuple(range(teacher_rank)))
    compact_optimizer = torch.optim.SGD(compact.parameters(), lr=lr)
    compact_risk_by_step: dict[int, float] = {0: half_mse(compact, test_x, test_y)}
    max_target_step = max(checkpoints) + horizon
    for step in range(1, max_target_step + 1):
        train_one_step(compact, compact_optimizer, train_x, train_y)
        if step in {c + horizon for c in checkpoints}:
            compact_risk_by_step[step] = half_mse(compact, test_x, test_y)

    dense_optimizer = torch.optim.SGD(dense.parameters(), lr=lr)
    records: list[dict] = []
    checkpoint_set = set(checkpoints)

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

            keep, immediate_reduced_selection = oracle_keep_indices(
                checkpoint_model,
                select_x,
                select_y,
                keep_rank=teacher_rank,
            )
            reduced = checkpoint_model.contracted_rank_copy(keep)
            dense_branch = copy.deepcopy(checkpoint_model)

            dense_wall = train_full_batch(dense_branch, train_x, train_y, steps=horizon, lr=lr)
            reduced_wall = train_full_batch(reduced, train_x, train_y, steps=horizon, lr=lr)

            dense_future_test = half_mse(dense_branch, test_x, test_y)
            reduced_future_test = half_mse(reduced, test_x, test_y)
            dense_future_selection = half_mse(dense_branch, select_x, select_y)
            reduced_future_selection = half_mse(reduced, select_x, select_y)
            compact_future_test = compact_risk_by_step[step + horizon]

            dense_params = parameter_count(dense_branch)
            reduced_params = parameter_count(reduced)
            records.append(
                {
                    "seed": int(seed),
                    "checkpoint": int(step),
                    "horizon_steps": int(horizon),
                    "effective_horizon": float(horizon * lr),
                    "current_dense_test_risk": current_dense_test,
                    "current_dense_selection_risk": current_dense_selection,
                    "oracle_keep_indices": list(keep),
                    "immediate_oracle_contraction_selection_damage": float(
                        immediate_reduced_selection - current_dense_selection
                    ),
                    "future_dense_test_risk": dense_future_test,
                    "future_reduced_test_risk": reduced_future_test,
                    "future_test_structural_damage": float(reduced_future_test - dense_future_test),
                    "future_dense_selection_risk": dense_future_selection,
                    "future_reduced_selection_risk": reduced_future_selection,
                    "future_selection_structural_damage": float(
                        reduced_future_selection - dense_future_selection
                    ),
                    "compact_from_start_test_risk_at_same_step": compact_future_test,
                    "wide_then_contract_minus_compact_from_start": float(
                        reduced_future_test - compact_future_test
                    ),
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
                    "reduced_continuation_wall_seconds": reduced_wall,
                }
            )

        if step < max(checkpoints):
            train_one_step(dense, dense_optimizer, train_x, train_y)

    return {
        "seed": int(seed),
        "initial_dense_parameters": parameter_count(initial_dense),
        "initial_compact_parameters": parameter_count(compact),
        "records": records,
    }


def summarize(results: list[dict]) -> dict:
    rows = [row for result in results for row in result["records"]]
    by_checkpoint: dict[str, dict] = {}
    for checkpoint in sorted({int(row["checkpoint"]) for row in rows}):
        group = [row for row in rows if int(row["checkpoint"]) == checkpoint]

        def mean(key: str) -> float:
            return float(np.mean([float(row[key]) for row in group]))

        by_checkpoint[str(checkpoint)] = {
            "n": len(group),
            "mean_future_test_structural_damage": mean("future_test_structural_damage"),
            "mean_wide_then_contract_minus_compact_from_start": mean(
                "wide_then_contract_minus_compact_from_start"
            ),
            "mean_bulk_target_mass_fraction": mean("bulk_target_mass_fraction"),
            "mean_bulk_gradient_energy_fraction": mean("bulk_gradient_energy_fraction"),
            "mean_bulk_fixed_geometry_value": mean("bulk_fixed_geometry_value"),
            "mean_eigengap": mean("eigengap"),
            "mean_parameter_step_fraction_saved": mean("parameter_step_fraction_saved"),
        }

    if len(rows) >= 3:
        damage = np.array([row["future_test_structural_damage"] for row in rows], dtype=float)
        metrics = {}
        for key in [
            "bulk_target_mass",
            "bulk_target_mass_fraction",
            "bulk_gradient_energy",
            "bulk_gradient_energy_fraction",
            "bulk_fixed_geometry_value",
            "eigengap",
        ]:
            x = np.array([row[key] for row in rows], dtype=float)
            if np.std(x) > 0 and np.std(damage) > 0:
                corr = float(np.corrcoef(x, damage)[0, 1])
            else:
                corr = 0.0
            metrics[key] = corr
    else:
        metrics = {}

    return {
        "by_checkpoint": by_checkpoint,
        "pearson_with_future_test_structural_damage": metrics,
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
    parser.add_argument("--checkpoints", type=int, nargs="+", default=[0, 20, 40, 60, 80, 120])
    parser.add_argument("--output", type=Path, default=Path("results/oracle_bulk_onset_sweep.json"))
    args = parser.parse_args()

    if args.teacher_rank >= args.student_rank:
        raise ValueError("teacher-rank must be smaller than student-rank")
    if sorted(set(args.checkpoints)) != args.checkpoints or args.checkpoints[0] < 0:
        raise ValueError("checkpoints must be strictly increasing nonnegative integers")

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
        )
        for seed in args.seeds
    ]

    payload = {
        "status": "development_oracle_onset_diagnostic",
        "warning": (
            "Structural subsets are selected by exhaustive target-aware search on an independent "
            "selection sample. This isolates WHEN from HOW and is not a deployable controller. "
            "The top-k tangent split is fixed to the known teacher rank and is also a synthetic "
            "oracle convention, not a learned bulk detector."
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
