from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch

from run_strict_oracle_onset_sweep import (
    half_mse,
    make_teacher_and_data,
    mean_ci95,
    oracle_keep_indices,
    parameter_count,
    train_full_batch,
    train_one_step,
)


def forward_matmul_macs_per_sample(
    *, input_dim: int, width1: int, width2: int, rank: int
) -> int:
    """Analytic multiply-accumulate proxy for the factorized MLP forward pass.

    Bias additions and nonlinearities are intentionally excluded. For fixed batch
    size, the same ratio is a useful first-order proxy for rank-dependent training
    compute because the dominant backward matmuls scale with the same dimensions.
    This is not reported as measured FLOPs.
    """
    return int(
        input_dim * width1
        + width1 * rank
        + rank * width2
        + width2
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
    (train_x, train_y), _, (select_x, select_y), (test_x, test_y) = blocks

    torch.manual_seed(30000 + int(seed))
    dense = __import__("neural_net.rank_factor", fromlist=["RankContractibleMLP"]).RankContractibleMLP(
        input_dim=input_dim,
        width1=width1,
        width2=width2,
        rank=student_rank,
    ).double()
    initial_dense = copy.deepcopy(dense)

    initial_keep, _ = oracle_keep_indices(
        initial_dense,
        select_x,
        select_y,
        keep_rank=teacher_rank,
    )
    compact = initial_dense.contracted_rank_copy(initial_keep)

    dense_macs = forward_matmul_macs_per_sample(
        input_dim=input_dim,
        width1=width1,
        width2=width2,
        rank=student_rank,
    )
    compact_macs = forward_matmul_macs_per_sample(
        input_dim=input_dim,
        width1=width1,
        width2=width2,
        rank=teacher_rank,
    )

    # Give the compact control at least as much analytic matmul compute as the
    # wide-then-contract path. ceil() intentionally favors the compact baseline.
    matched_steps = {
        checkpoint: int(
            math.ceil((dense_macs * checkpoint + compact_macs * horizon) / compact_macs)
        )
        for checkpoint in checkpoints
    }
    max_compact_steps = max(matched_steps.values())

    compact_optimizer = torch.optim.SGD(compact.parameters(), lr=lr)
    compact_risk_by_step: dict[int, float] = {0: half_mse(compact, test_x, test_y)}
    target_compact_steps = set(matched_steps.values())
    for step in range(1, max_compact_steps + 1):
        train_one_step(compact, compact_optimizer, train_x, train_y)
        if step in target_compact_steps:
            compact_risk_by_step[step] = half_mse(compact, test_x, test_y)

    dense_optimizer = torch.optim.SGD(dense.parameters(), lr=lr)
    checkpoint_set = set(checkpoints)
    records: list[dict] = []

    for step in range(max(checkpoints) + 1):
        if step in checkpoint_set:
            checkpoint_model = copy.deepcopy(dense)
            oracle_keep, _ = oracle_keep_indices(
                checkpoint_model,
                select_x,
                select_y,
                keep_rank=teacher_rank,
            )
            reduced = checkpoint_model.contracted_rank_copy(oracle_keep)
            dense_branch = copy.deepcopy(checkpoint_model)

            train_full_batch(reduced, train_x, train_y, steps=horizon, lr=lr)
            train_full_batch(dense_branch, train_x, train_y, steps=horizon, lr=lr)

            reduced_risk = half_mse(reduced, test_x, test_y)
            dense_risk = half_mse(dense_branch, test_x, test_y)
            compact_steps = matched_steps[step]
            compact_equal_compute_risk = compact_risk_by_step[compact_steps]

            wide_compute = int(dense_macs * step + compact_macs * horizon)
            compact_compute = int(compact_macs * compact_steps)
            records.append(
                {
                    "seed": int(seed),
                    "checkpoint": int(step),
                    "horizon_steps": int(horizon),
                    "oracle_keep_indices": list(oracle_keep),
                    "future_dense_test_risk": float(dense_risk),
                    "future_wide_then_contract_test_risk": float(reduced_risk),
                    "same_step_structural_damage": float(reduced_risk - dense_risk),
                    "compact_equal_compute_steps": int(compact_steps),
                    "compact_extra_steps_vs_equal_step_count": int(compact_steps - (step + horizon)),
                    "compact_equal_compute_test_risk": float(compact_equal_compute_risk),
                    "equal_compute_gap_wide_minus_compact": float(
                        reduced_risk - compact_equal_compute_risk
                    ),
                    "wide_beats_compact_at_equal_compute": bool(
                        reduced_risk < compact_equal_compute_risk
                    ),
                    "dense_forward_matmul_macs_per_sample": int(dense_macs),
                    "compact_forward_matmul_macs_per_sample": int(compact_macs),
                    "wide_path_matmul_mac_proxy": wide_compute,
                    "compact_path_matmul_mac_proxy": compact_compute,
                    "compact_compute_advantage_fraction": float(
                        compact_compute / wide_compute - 1.0 if wide_compute > 0 else 0.0
                    ),
                    "dense_parameters": parameter_count(dense_branch),
                    "reduced_parameters": parameter_count(reduced),
                }
            )

        if step < max(checkpoints):
            train_one_step(dense, dense_optimizer, train_x, train_y)

    return {
        "seed": int(seed),
        "initial_oracle_keep_indices": list(initial_keep),
        "dense_forward_matmul_macs_per_sample": int(dense_macs),
        "compact_forward_matmul_macs_per_sample": int(compact_macs),
        "rank_dependent_forward_mac_reduction_fraction": float(1.0 - compact_macs / dense_macs),
        "records": records,
    }


def summarize(results: list[dict]) -> dict:
    rows = [row for result in results for row in result["records"]]
    by_checkpoint: dict[str, dict] = {}
    for checkpoint in sorted({int(row["checkpoint"]) for row in rows}):
        group = [row for row in rows if int(row["checkpoint"]) == checkpoint]
        gaps = [float(row["equal_compute_gap_wide_minus_compact"]) for row in group]
        damage = [float(row["same_step_structural_damage"]) for row in group]
        by_checkpoint[str(checkpoint)] = {
            "equal_compute_gap_wide_minus_compact": mean_ci95(gaps),
            "same_step_structural_damage": mean_ci95(damage),
            "wide_win_fraction_equal_compute": float(
                np.mean([bool(row["wide_beats_compact_at_equal_compute"]) for row in group])
            ),
            "mean_compact_equal_compute_steps": float(
                np.mean([int(row["compact_equal_compute_steps"]) for row in group])
            ),
            "mean_compact_extra_steps_vs_equal_step_count": float(
                np.mean([int(row["compact_extra_steps_vs_equal_step_count"]) for row in group])
            ),
        }

    nonzero_rows = [row for row in rows if int(row["checkpoint"]) > 0]
    return {
        "by_checkpoint": by_checkpoint,
        "all_post_initialization": {
            "equal_compute_gap_wide_minus_compact": mean_ci95(
                [float(row["equal_compute_gap_wide_minus_compact"]) for row in nonzero_rows]
            ),
            "wide_win_fraction_equal_compute": float(
                np.mean([bool(row["wide_beats_compact_at_equal_compute"]) for row in nonzero_rows])
            ) if nonzero_rows else 0.0,
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
    parser.add_argument("--checkpoints", type=int, nargs="+", default=[0, 20, 40, 60, 80, 100, 120])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/equal_compute_oracle_sweep.json"),
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
        )
        for seed in args.seeds
    ]

    payload = {
        "status": "oracle_equal_compute_falsification_test",
        "warning": (
            "Compute matching uses an analytic factorized-matmul MAC proxy, not measured hardware FLOPs. "
            "The compact control receives ceil-matched compute, slightly favoring it. Structural subset "
            "selection is still target-aware oracle selection on a held-out sample."
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
