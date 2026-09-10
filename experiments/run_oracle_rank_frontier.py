from __future__ import annotations

import argparse
import copy
import json
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
    parameter_count,
    train_full_batch,
    train_one_step,
)


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
    (train_x, train_y), (spectral_x, spectral_y), _, (test_x, test_y) = blocks

    torch.manual_seed(30000 + int(seed))
    dense = RankContractibleMLP(
        input_dim=input_dim,
        width1=width1,
        width2=width2,
        rank=student_rank,
    ).double()
    initial_dense = copy.deepcopy(dense)

    # Fair compact-from-start controls.  Each rank-r model is exactly the rank-r
    # SVD recompression of the identical initial wide model.
    target_steps = {int(c + horizon) for c in checkpoints}
    max_target_step = max(target_steps)
    compact_risk_by_rank_step: dict[int, dict[int, float]] = {}
    for rank in range(1, student_rank):
        compact = initial_dense.svd_contracted_rank_copy(rank)
        optimizer = torch.optim.SGD(compact.parameters(), lr=lr)
        compact_risk_by_rank_step[rank] = {0: half_mse(compact, test_x, test_y)}
        for step in range(1, max_target_step + 1):
            train_one_step(compact, optimizer, train_x, train_y)
            if step in target_steps:
                compact_risk_by_rank_step[rank][step] = half_mse(compact, test_x, test_y)

    dense_optimizer = torch.optim.SGD(dense.parameters(), lr=lr)
    checkpoint_set = set(checkpoints)
    checkpoint_records: list[dict] = []

    for step in range(max(checkpoints) + 1):
        if step in checkpoint_set:
            checkpoint_model = copy.deepcopy(dense)
            tangent = empirical_tangent_spectrum(checkpoint_model, spectral_x, spectral_y)
            bulk = bulk_state_from_tangent_spectrum(
                tangent,
                signal_rank=teacher_rank,
                horizon=float(horizon * lr),
            )

            dense_branch = copy.deepcopy(checkpoint_model)
            train_full_batch(dense_branch, train_x, train_y, steps=horizon, lr=lr)
            dense_future_test = half_mse(dense_branch, test_x, test_y)
            tolerance = float(max(damage_abs_tol, damage_rel_tol * dense_future_test))

            rank_records: list[dict] = []
            for rank in range(1, student_rank):
                compact_branch = checkpoint_model.svd_contracted_rank_copy(rank)
                immediate_test = half_mse(compact_branch, test_x, test_y)
                singular_tail = float(
                    checkpoint_model.hidden2.singular_value_tail_fraction(rank).detach().cpu()
                )
                train_full_batch(compact_branch, train_x, train_y, steps=horizon, lr=lr)
                future_test = half_mse(compact_branch, test_x, test_y)
                damage = float(future_test - dense_future_test)
                compact_start_risk = compact_risk_by_rank_step[rank][int(step + horizon)]
                rank_records.append(
                    {
                        "rank": int(rank),
                        "singular_value_tail_fraction": singular_tail,
                        "immediate_test_risk": float(immediate_test),
                        "future_test_risk": float(future_test),
                        "future_structural_damage": damage,
                        "safe": bool(damage <= tolerance),
                        "compact_from_start_same_steps_test_risk": float(compact_start_risk),
                        "width_path_value_same_steps": float(compact_start_risk - future_test),
                        "wide_then_contract_beats_compact_same_steps": bool(future_test < compact_start_risk),
                        "reduced_parameters": parameter_count(compact_branch),
                        "parameter_fraction_removed": float(
                            1.0 - parameter_count(compact_branch) / parameter_count(checkpoint_model)
                        ),
                    }
                )

            # Dense rank is always a feasible safe endpoint with zero structural damage.
            rank_records.append(
                {
                    "rank": int(student_rank),
                    "singular_value_tail_fraction": 0.0,
                    "immediate_test_risk": half_mse(checkpoint_model, test_x, test_y),
                    "future_test_risk": float(dense_future_test),
                    "future_structural_damage": 0.0,
                    "safe": True,
                    "compact_from_start_same_steps_test_risk": None,
                    "width_path_value_same_steps": None,
                    "wide_then_contract_beats_compact_same_steps": None,
                    "reduced_parameters": parameter_count(checkpoint_model),
                    "parameter_fraction_removed": 0.0,
                }
            )

            safe_ranks = [int(row["rank"]) for row in rank_records if bool(row["safe"])]
            oracle_rank = min(safe_ranks)
            damage_sequence = np.asarray(
                [float(row["future_structural_damage"]) for row in rank_records], dtype=float
            )
            monotone_violations = int(np.sum(np.diff(damage_sequence) > 1e-10))

            checkpoint_records.append(
                {
                    "seed": int(seed),
                    "checkpoint": int(step),
                    "horizon_steps": int(horizon),
                    "effective_horizon": float(horizon * lr),
                    "dense_future_test_risk": float(dense_future_test),
                    "damage_tolerance": tolerance,
                    "oracle_minimal_safe_rank": int(oracle_rank),
                    "oracle_parameter_fraction_removed": float(
                        next(
                            row["parameter_fraction_removed"]
                            for row in rank_records
                            if int(row["rank"]) == oracle_rank
                        )
                    ),
                    "damage_monotonicity_violations_across_rank": monotone_violations,
                    "signal_floor": bulk.signal_floor,
                    "bulk_edge": bulk.bulk_edge,
                    "eigengap": bulk.eigengap,
                    "bulk_target_mass": bulk.bulk_target_mass,
                    "bulk_target_mass_fraction": bulk.bulk_target_mass_fraction,
                    "bulk_gradient_energy": bulk.bulk_gradient_energy,
                    "bulk_gradient_energy_fraction": bulk.bulk_gradient_energy_fraction,
                    "bulk_fixed_geometry_value": bulk.bulk_fixed_geometry_value,
                    "rank_records": rank_records,
                }
            )

        if step < max(checkpoints):
            train_one_step(dense, dense_optimizer, train_x, train_y)

    oracle_ranks = [int(row["oracle_minimal_safe_rank"]) for row in checkpoint_records]
    # Smallest rank that is safe to adopt irreversibly at checkpoint t if the
    # architecture is not allowed to grow again: max future oracle rank.
    irreversible_frontier = [max(oracle_ranks[i:]) for i in range(len(oracle_ranks))]
    first_real_contraction = next(
        (
            int(checkpoint_records[i]["checkpoint"])
            for i, rank in enumerate(irreversible_frontier)
            if rank < student_rank
        ),
        None,
    )
    first_teacher_rank = next(
        (
            int(checkpoint_records[i]["checkpoint"])
            for i, rank in enumerate(irreversible_frontier)
            if rank <= teacher_rank
        ),
        None,
    )
    for row, irreversible_rank in zip(checkpoint_records, irreversible_frontier):
        row["oracle_irreversible_safe_rank"] = int(irreversible_rank)

    return {
        "seed": int(seed),
        "first_irreversible_contraction_checkpoint": first_real_contraction,
        "first_irreversible_teacher_rank_checkpoint": first_teacher_rank,
        "oracle_rank_sequence": oracle_ranks,
        "oracle_irreversible_rank_sequence": irreversible_frontier,
        "checkpoints": checkpoint_records,
    }


def summarize(results: list[dict], student_rank: int, teacher_rank: int) -> dict:
    rows = [row for result in results for row in result["checkpoints"]]
    checkpoints = sorted({int(row["checkpoint"]) for row in rows})
    by_checkpoint: dict[str, dict] = {}
    for checkpoint in checkpoints:
        group = [row for row in rows if int(row["checkpoint"]) == checkpoint]
        oracle = [float(row["oracle_minimal_safe_rank"]) for row in group]
        irreversible = [float(row["oracle_irreversible_safe_rank"]) for row in group]
        by_checkpoint[str(checkpoint)] = {
            "oracle_minimal_safe_rank": mean_ci95(oracle),
            "oracle_irreversible_safe_rank": mean_ci95(irreversible),
            "fraction_any_contraction_safe": float(
                np.mean([float(x) < student_rank for x in oracle])
            ),
            "fraction_teacher_rank_safe": float(
                np.mean([float(x) <= teacher_rank for x in oracle])
            ),
            "fraction_irreversible_contraction_safe": float(
                np.mean([float(x) < student_rank for x in irreversible])
            ),
            "fraction_irreversible_teacher_rank_safe": float(
                np.mean([float(x) <= teacher_rank for x in irreversible])
            ),
            "oracle_parameter_fraction_removed": mean_ci95(
                [float(row["oracle_parameter_fraction_removed"]) for row in group]
            ),
            "bulk_fixed_geometry_value": mean_ci95(
                [float(row["bulk_fixed_geometry_value"]) for row in group]
            ),
            "eigengap": mean_ci95([float(row["eigengap"]) for row in group]),
        }

    first_contract = [
        result["first_irreversible_contraction_checkpoint"]
        for result in results
        if result["first_irreversible_contraction_checkpoint"] is not None
    ]
    first_teacher = [
        result["first_irreversible_teacher_rank_checkpoint"]
        for result in results
        if result["first_irreversible_teacher_rank_checkpoint"] is not None
    ]

    return {
        "by_checkpoint": by_checkpoint,
        "first_irreversible_contraction": {
            "resolved_fraction": float(len(first_contract) / len(results)) if results else 0.0,
            "checkpoints": first_contract,
            "mean": float(np.mean(first_contract)) if first_contract else None,
            "median": float(np.median(first_contract)) if first_contract else None,
        },
        "first_irreversible_teacher_rank": {
            "resolved_fraction": float(len(first_teacher) / len(results)) if results else 0.0,
            "checkpoints": first_teacher,
            "mean": float(np.mean(first_teacher)) if first_teacher else None,
            "median": float(np.median(first_teacher)) if first_teacher else None,
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
    parser.add_argument("--checkpoints", type=int, nargs="+", default=[0, 20, 40, 60, 80, 100, 120])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/oracle_rank_frontier.json"),
    )
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
            damage_abs_tol=args.damage_abs_tol,
            damage_rel_tol=args.damage_rel_tol,
        )
        for seed in args.seeds
    ]

    payload = {
        "status": "oracle_svd_rank_frontier",
        "warning": (
            "The contraction itself is target-free truncated SVD, but the minimal safe rank is defined "
            "retrospectively using future test risk. This is ground truth for studying when and how far "
            "a network could have been contracted, not a deployable rank controller."
        ),
        "config": vars(args) | {"output": str(args.output)},
        "runs": runs,
        "summary": summarize(runs, args.student_rank, args.teacher_rank),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
