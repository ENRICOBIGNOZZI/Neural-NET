from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch

from neural_net.bulk_spectrum import bulk_state_from_tangent_spectrum
from neural_net.rank_factor import RankContractibleMLP
from neural_net.torch_probe import empirical_tangent_spectrum

from run_strict_oracle_onset_sweep import (
    half_mse,
    make_teacher_and_data,
    mean_ci95,
    oracle_keep_indices,
    parameter_count,
    random_keep_indices,
    train_full_batch,
    train_one_step,
)


def relative_weight_error_sq(
    full: RankContractibleMLP,
    compact: RankContractibleMLP,
    eps: float = 1e-18,
) -> float:
    full_weight = full.hidden2.effective_weight().detach()
    compact_weight = compact.hidden2.effective_weight().detach()
    denom = float(torch.sum(full_weight.square()).cpu())
    if denom <= eps:
        return 0.0
    return float((torch.sum((full_weight - compact_weight).square()) / denom).cpu())


def run_seed(
    seed: int,
    *,
    input_dim: int,
    width1: int,
    width2: int,
    teacher_rank: int,
    student_rank: int,
    target_ranks: list[int],
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

    target_steps = {int(c + horizon) for c in checkpoints}
    max_target_step = max(target_steps)
    compact_risks: dict[int, dict[int, float]] = {}
    initial_svd_risks: dict[int, float] = {}
    for target_rank in target_ranks:
        compact = initial_dense.svd_contracted_rank_copy(target_rank)
        initial_svd_risks[target_rank] = half_mse(compact, test_x, test_y)
        optimizer = torch.optim.SGD(compact.parameters(), lr=lr)
        compact_risks[target_rank] = {0: half_mse(compact, test_x, test_y)}
        for step in range(1, max_target_step + 1):
            train_one_step(compact, optimizer, train_x, train_y)
            if step in target_steps:
                compact_risks[target_rank][step] = half_mse(compact, test_x, test_y)

    dense_optimizer = torch.optim.SGD(dense.parameters(), lr=lr)
    checkpoint_set = set(checkpoints)
    records: list[dict] = []

    for step in range(max(checkpoints) + 1):
        if step in checkpoint_set:
            checkpoint_model = copy.deepcopy(dense)
            dense_now_test = half_mse(checkpoint_model, test_x, test_y)
            dense_now_selection = half_mse(checkpoint_model, select_x, select_y)

            tangent = empirical_tangent_spectrum(checkpoint_model, spectral_x, spectral_y)
            bulk = bulk_state_from_tangent_spectrum(
                tangent,
                signal_rank=teacher_rank,
                horizon=float(horizon * lr),
            )

            dense_branch = copy.deepcopy(checkpoint_model)
            train_full_batch(dense_branch, train_x, train_y, steps=horizon, lr=lr)
            dense_future_test = half_mse(dense_branch, test_x, test_y)

            for target_rank in target_ranks:
                svd_branch = checkpoint_model.svd_contracted_rank_copy(target_rank)
                oracle_keep, oracle_now_selection = oracle_keep_indices(
                    checkpoint_model,
                    select_x,
                    select_y,
                    keep_rank=target_rank,
                )
                oracle_branch = checkpoint_model.contracted_rank_copy(oracle_keep)
                random_keep = random_keep_indices(seed, step + 10000 * target_rank, student_rank, target_rank)
                random_branch = checkpoint_model.contracted_rank_copy(random_keep)

                svd_now_test = half_mse(svd_branch, test_x, test_y)
                svd_now_selection = half_mse(svd_branch, select_x, select_y)
                oracle_now_test = half_mse(oracle_branch, test_x, test_y)
                random_now_test = half_mse(random_branch, test_x, test_y)

                singular_tail = float(
                    checkpoint_model.hidden2.singular_value_tail_fraction(target_rank).detach().cpu()
                )
                svd_weight_error = relative_weight_error_sq(checkpoint_model, svd_branch)
                oracle_weight_error = relative_weight_error_sq(checkpoint_model, oracle_branch)
                random_weight_error = relative_weight_error_sq(checkpoint_model, random_branch)

                train_full_batch(svd_branch, train_x, train_y, steps=horizon, lr=lr)
                train_full_batch(oracle_branch, train_x, train_y, steps=horizon, lr=lr)
                train_full_batch(random_branch, train_x, train_y, steps=horizon, lr=lr)

                svd_future_test = half_mse(svd_branch, test_x, test_y)
                oracle_future_test = half_mse(oracle_branch, test_x, test_y)
                random_future_test = half_mse(random_branch, test_x, test_y)
                compact_future_test = compact_risks[target_rank][int(step + horizon)]

                svd_damage = float(svd_future_test - dense_future_test)
                tolerance = float(max(damage_abs_tol, damage_rel_tol * dense_future_test))
                records.append(
                    {
                        "seed": int(seed),
                        "checkpoint": int(step),
                        "target_rank": int(target_rank),
                        "horizon_steps": int(horizon),
                        "effective_horizon": float(horizon * lr),
                        "current_dense_test_risk": dense_now_test,
                        "current_dense_selection_risk": dense_now_selection,
                        "current_svd_test_risk": svd_now_test,
                        "current_oracle_subset_test_risk": oracle_now_test,
                        "current_random_subset_test_risk": random_now_test,
                        "immediate_svd_test_damage": float(svd_now_test - dense_now_test),
                        "immediate_svd_selection_damage": float(svd_now_selection - dense_now_selection),
                        "immediate_oracle_subset_selection_damage": float(
                            oracle_now_selection - dense_now_selection
                        ),
                        "oracle_keep_indices": list(oracle_keep),
                        "random_keep_indices": list(random_keep),
                        "singular_value_tail_fraction": singular_tail,
                        "svd_relative_weight_error_sq": svd_weight_error,
                        "oracle_subset_relative_weight_error_sq": oracle_weight_error,
                        "random_subset_relative_weight_error_sq": random_weight_error,
                        "future_dense_test_risk": dense_future_test,
                        "future_svd_test_risk": svd_future_test,
                        "future_oracle_subset_test_risk": oracle_future_test,
                        "future_random_subset_test_risk": random_future_test,
                        "future_svd_structural_damage": svd_damage,
                        "future_oracle_subset_structural_damage": float(
                            oracle_future_test - dense_future_test
                        ),
                        "future_random_subset_structural_damage": float(
                            random_future_test - dense_future_test
                        ),
                        "damage_tolerance": tolerance,
                        "safe_svd_contraction": bool(svd_damage <= tolerance),
                        "compact_svd_from_start_test_risk_at_same_step": compact_future_test,
                        "svd_width_path_value": float(compact_future_test - svd_future_test),
                        "svd_beats_compact_same_steps": bool(svd_future_test < compact_future_test),
                        "svd_gain_vs_oracle_component_subset": float(
                            oracle_future_test - svd_future_test
                        ),
                        "svd_gain_vs_random_component_subset": float(
                            random_future_test - svd_future_test
                        ),
                        "signal_floor": bulk.signal_floor,
                        "bulk_edge": bulk.bulk_edge,
                        "eigengap": bulk.eigengap,
                        "bulk_target_mass": bulk.bulk_target_mass,
                        "bulk_target_mass_fraction": bulk.bulk_target_mass_fraction,
                        "bulk_gradient_energy": bulk.bulk_gradient_energy,
                        "bulk_gradient_energy_fraction": bulk.bulk_gradient_energy_fraction,
                        "bulk_fixed_geometry_value": bulk.bulk_fixed_geometry_value,
                        "dense_parameters": parameter_count(checkpoint_model),
                        "reduced_parameters": parameter_count(svd_branch),
                        "parameter_fraction_removed": float(
                            1.0 - parameter_count(svd_branch) / parameter_count(checkpoint_model)
                        ),
                    }
                )

        if step < max(checkpoints):
            train_one_step(dense, dense_optimizer, train_x, train_y)

    persistent_onsets: dict[str, int | None] = {}
    for target_rank in target_ranks:
        rank_rows = [row for row in records if int(row["target_rank"]) == target_rank]
        onset = None
        for i, row in enumerate(rank_rows):
            if row["safe_svd_contraction"] and all(r["safe_svd_contraction"] for r in rank_rows[i:]):
                onset = int(row["checkpoint"])
                break
        persistent_onsets[str(target_rank)] = onset

    return {
        "seed": int(seed),
        "initial_dense_parameters": parameter_count(initial_dense),
        "initial_svd_test_risks": {str(k): float(v) for k, v in initial_svd_risks.items()},
        "persistent_safe_svd_onset_by_rank": persistent_onsets,
        "records": records,
    }


def summarize(results: list[dict]) -> dict:
    rows = [row for result in results for row in result["records"]]
    target_ranks = sorted({int(row["target_rank"]) for row in rows})
    checkpoints = sorted({int(row["checkpoint"]) for row in rows})
    by_rank: dict[str, dict] = {}

    for target_rank in target_ranks:
        rank_rows = [row for row in rows if int(row["target_rank"]) == target_rank]
        by_checkpoint: dict[str, dict] = {}
        for checkpoint in checkpoints:
            group = [row for row in rank_rows if int(row["checkpoint"]) == checkpoint]
            by_checkpoint[str(checkpoint)] = {
                "future_svd_structural_damage": mean_ci95(
                    [float(row["future_svd_structural_damage"]) for row in group]
                ),
                "future_oracle_subset_structural_damage": mean_ci95(
                    [float(row["future_oracle_subset_structural_damage"]) for row in group]
                ),
                "singular_value_tail_fraction": mean_ci95(
                    [float(row["singular_value_tail_fraction"]) for row in group]
                ),
                "svd_width_path_value": mean_ci95(
                    [float(row["svd_width_path_value"]) for row in group]
                ),
                "svd_gain_vs_oracle_component_subset": mean_ci95(
                    [float(row["svd_gain_vs_oracle_component_subset"]) for row in group]
                ),
                "safe_svd_fraction": float(
                    np.mean([bool(row["safe_svd_contraction"]) for row in group])
                ),
                "svd_same_step_win_fraction": float(
                    np.mean([bool(row["svd_beats_compact_same_steps"]) for row in group])
                ),
                "parameter_fraction_removed": float(
                    np.mean([float(row["parameter_fraction_removed"]) for row in group])
                ),
            }

        damage = np.asarray(
            [float(row["future_svd_structural_damage"]) for row in rank_rows], dtype=float
        )
        correlations: dict[str, float] = {}
        for key in [
            "singular_value_tail_fraction",
            "bulk_target_mass",
            "bulk_gradient_energy",
            "bulk_fixed_geometry_value",
            "eigengap",
        ]:
            x = np.asarray([float(row[key]) for row in rank_rows], dtype=float)
            correlations[key] = (
                float(np.corrcoef(x, damage)[0, 1])
                if x.size >= 3 and np.std(x) > 0.0 and np.std(damage) > 0.0
                else 0.0
            )

        onsets = [
            result["persistent_safe_svd_onset_by_rank"][str(target_rank)]
            for result in results
            if result["persistent_safe_svd_onset_by_rank"][str(target_rank)] is not None
        ]
        by_rank[str(target_rank)] = {
            "by_checkpoint": by_checkpoint,
            "pearson_with_future_svd_damage": correlations,
            "persistent_safe_onset": {
                "resolved_fraction": float(len(onsets) / len(results)) if results else 0.0,
                "onsets": onsets,
                "mean": float(np.mean(onsets)) if onsets else None,
                "median": float(np.median(onsets)) if onsets else None,
            },
        }

    return {"by_rank": by_rank}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--input-dim", type=int, default=5)
    parser.add_argument("--width1", type=int, default=12)
    parser.add_argument("--width2", type=int, default=12)
    parser.add_argument("--teacher-rank", type=int, default=2)
    parser.add_argument("--student-rank", type=int, default=8)
    parser.add_argument("--target-ranks", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--train-n", type=int, default=256)
    parser.add_argument("--spectral-n", type=int, default=48)
    parser.add_argument("--selection-n", type=int, default=512)
    parser.add_argument("--test-n", type=int, default=2048)
    parser.add_argument("--noise-std", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--horizon", type=int, default=40)
    parser.add_argument("--damage-abs-tol", type=float, default=1e-3)
    parser.add_argument("--damage-rel-tol", type=float, default=0.05)
    parser.add_argument("--checkpoints", type=int, nargs="+", default=[0, 20, 40, 60, 80, 100, 120])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/svd_structural_bridge_sweep.json"),
    )
    args = parser.parse_args()

    if args.teacher_rank >= args.student_rank:
        raise ValueError("teacher-rank must be smaller than student-rank")
    if any(rank < 1 or rank >= args.student_rank for rank in args.target_ranks):
        raise ValueError("target ranks must lie between 1 and student_rank - 1")
    if len(set(args.target_ranks)) != len(args.target_ranks):
        raise ValueError("target ranks must be unique")
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
            target_ranks=args.target_ranks,
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
        "status": "svd_structural_bridge_diagnostic",
        "warning": (
            "SVD recompression is target-agnostic and optimal only for the contractible weight matrix in "
            "Frobenius norm, not for network risk. The raw component-subset comparator remains target-aware "
            "oracle selection. This experiment tests the structural bridge rather than a deployable controller."
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
