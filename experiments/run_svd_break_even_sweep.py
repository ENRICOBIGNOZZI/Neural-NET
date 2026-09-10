from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch

from neural_net.rank_factor import RankContractibleMLP

from run_equal_compute_oracle_sweep import forward_matmul_macs_per_sample
from run_strict_oracle_onset_sweep import (
    half_mse,
    make_teacher_and_data,
    mean_ci95,
    parameter_count,
    train_full_batch,
    train_one_step,
)


def _train_risk_curve(
    model: torch.nn.Module,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    *,
    lr: float,
    target_steps: set[int],
) -> dict[int, float]:
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    risks = {0: half_mse(model, test_x, test_y)}
    if not target_steps:
        return risks
    for step in range(1, max(target_steps) + 1):
        train_one_step(model, optimizer, train_x, train_y)
        if step in target_steps:
            risks[step] = half_mse(model, test_x, test_y)
    return risks


def _seed_aggregated_ci(rows: list[dict], key: str) -> dict:
    by_seed: dict[int, list[float]] = {}
    for row in rows:
        by_seed.setdefault(int(row["seed"]), []).append(float(row[key]))
    seed_means = [float(np.mean(values)) for _, values in sorted(by_seed.items())]
    out = mean_ci95(seed_means)
    out["aggregation"] = "mean within seed over post-initialization checkpoints, then CI across seeds"
    return out


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
    (train_x, train_y), _, _, (test_x, test_y) = blocks

    torch.manual_seed(30000 + int(seed))
    wide = RankContractibleMLP(
        input_dim=input_dim,
        width1=width1,
        width2=width2,
        rank=student_rank,
    ).double()
    initial_wide = copy.deepcopy(wide)

    wide_macs = forward_matmul_macs_per_sample(
        input_dim=input_dim,
        width1=width1,
        width2=width2,
        rank=student_rank,
    )
    compact_macs = {
        rank: forward_matmul_macs_per_sample(
            input_dim=input_dim,
            width1=width1,
            width2=width2,
            rank=rank,
        )
        for rank in target_ranks
    }
    equal_compute_steps = {
        rank: {
            checkpoint: int(
                math.ceil(
                    (wide_macs * checkpoint + compact_macs[rank] * horizon)
                    / compact_macs[rank]
                )
            )
            for checkpoint in checkpoints
        }
        for rank in target_ranks
    }
    same_steps = {checkpoint: int(checkpoint + horizon) for checkpoint in checkpoints}

    matched_risks: dict[int, dict[int, float]] = {}
    native_risks: dict[int, dict[int, float]] = {}
    for rank in target_ranks:
        targets = set(equal_compute_steps[rank].values()) | set(same_steps.values())

        matched = initial_wide.svd_contracted_rank_copy(rank)
        matched_risks[rank] = _train_risk_curve(
            matched,
            train_x,
            train_y,
            test_x,
            test_y,
            lr=lr,
            target_steps=targets,
        )

        torch.manual_seed(40000 + 100 * int(seed) + int(rank))
        native = RankContractibleMLP(
            input_dim=input_dim,
            width1=width1,
            width2=width2,
            rank=rank,
        ).double()
        native_risks[rank] = _train_risk_curve(
            native,
            train_x,
            train_y,
            test_x,
            test_y,
            lr=lr,
            target_steps=targets,
        )

    optimizer = torch.optim.SGD(wide.parameters(), lr=lr)
    checkpoint_set = set(checkpoints)
    records: list[dict] = []

    for step in range(max(checkpoints) + 1):
        if step in checkpoint_set:
            checkpoint_model = copy.deepcopy(wide)
            dense_branch = copy.deepcopy(checkpoint_model)
            train_full_batch(dense_branch, train_x, train_y, steps=horizon, lr=lr)
            dense_future_risk = half_mse(dense_branch, test_x, test_y)

            for rank in target_ranks:
                reduced = checkpoint_model.svd_contracted_rank_copy(rank)
                singular_tail = float(
                    checkpoint_model.hidden2.singular_value_tail_fraction(rank).detach().cpu()
                )
                train_full_batch(reduced, train_x, train_y, steps=horizon, lr=lr)
                wide_then_svd_risk = half_mse(reduced, test_x, test_y)

                same_k = same_steps[step]
                equal_k = equal_compute_steps[rank][step]
                matched_same = matched_risks[rank][same_k]
                matched_equal = matched_risks[rank][equal_k]
                native_same = native_risks[rank][same_k]
                native_equal = native_risks[rank][equal_k]

                matched_discovery_advantage = matched_same - wide_then_svd_risk
                matched_compute_opportunity_gain = matched_same - matched_equal
                matched_equal_gap = wide_then_svd_risk - matched_equal
                matched_identity_error = matched_equal_gap - (
                    matched_compute_opportunity_gain - matched_discovery_advantage
                )

                native_discovery_advantage = native_same - wide_then_svd_risk
                native_compute_opportunity_gain = native_same - native_equal
                native_equal_gap = wide_then_svd_risk - native_equal
                native_identity_error = native_equal_gap - (
                    native_compute_opportunity_gain - native_discovery_advantage
                )

                wide_compute = int(wide_macs * step + compact_macs[rank] * horizon)
                compact_compute = int(compact_macs[rank] * equal_k)

                records.append(
                    {
                        "seed": int(seed),
                        "checkpoint": int(step),
                        "target_rank": int(rank),
                        "horizon_steps": int(horizon),
                        "same_step_compact_steps": int(same_k),
                        "equal_compute_compact_steps": int(equal_k),
                        "compact_extra_steps": int(equal_k - same_k),
                        "wide_to_compact_cost_ratio": float(wide_macs / compact_macs[rank]),
                        "singular_value_tail_fraction": singular_tail,
                        "future_dense_test_risk": float(dense_future_risk),
                        "future_wide_then_svd_test_risk": float(wide_then_svd_risk),
                        "same_step_structural_damage_vs_dense": float(
                            wide_then_svd_risk - dense_future_risk
                        ),
                        "matched_compact_same_step_test_risk": float(matched_same),
                        "matched_compact_equal_compute_test_risk": float(matched_equal),
                        "matched_same_step_gap_wide_minus_compact": float(
                            wide_then_svd_risk - matched_same
                        ),
                        "matched_discovery_advantage": float(matched_discovery_advantage),
                        "matched_compute_opportunity_gain": float(matched_compute_opportunity_gain),
                        "matched_equal_compute_gap_wide_minus_compact": float(matched_equal_gap),
                        "matched_break_even_margin": float(
                            matched_discovery_advantage - matched_compute_opportunity_gain
                        ),
                        "matched_identity_error": float(matched_identity_error),
                        "wide_beats_matched_same_steps": bool(wide_then_svd_risk < matched_same),
                        "wide_beats_matched_equal_compute": bool(wide_then_svd_risk < matched_equal),
                        "native_compact_same_step_test_risk": float(native_same),
                        "native_compact_equal_compute_test_risk": float(native_equal),
                        "native_same_step_gap_wide_minus_compact": float(
                            wide_then_svd_risk - native_same
                        ),
                        "native_discovery_advantage": float(native_discovery_advantage),
                        "native_compute_opportunity_gain": float(native_compute_opportunity_gain),
                        "native_equal_compute_gap_wide_minus_compact": float(native_equal_gap),
                        "native_break_even_margin": float(
                            native_discovery_advantage - native_compute_opportunity_gain
                        ),
                        "native_identity_error": float(native_identity_error),
                        "wide_beats_native_same_steps": bool(wide_then_svd_risk < native_same),
                        "wide_beats_native_equal_compute": bool(wide_then_svd_risk < native_equal),
                        "wide_path_matmul_mac_proxy": wide_compute,
                        "compact_path_matmul_mac_proxy": compact_compute,
                        "compact_compute_advantage_fraction": float(
                            compact_compute / wide_compute - 1.0 if wide_compute > 0 else 0.0
                        ),
                        "dense_parameters": parameter_count(checkpoint_model),
                        "reduced_parameters": parameter_count(reduced),
                        "parameter_fraction_removed": float(
                            1.0 - parameter_count(reduced) / parameter_count(checkpoint_model)
                        ),
                    }
                )

        if step < max(checkpoints):
            train_one_step(wide, optimizer, train_x, train_y)

    max_identity_error = max(
        max(abs(float(row["matched_identity_error"])), abs(float(row["native_identity_error"])))
        for row in records
    )
    if max_identity_error > 1e-12:
        raise AssertionError(f"break-even identity failed: max error={max_identity_error}")

    return {
        "seed": int(seed),
        "wide_forward_matmul_macs_per_sample": int(wide_macs),
        "compact_forward_matmul_macs_per_sample": {
            str(rank): int(macs) for rank, macs in compact_macs.items()
        },
        "records": records,
    }


def _checkpoint_summary(group: list[dict]) -> dict:
    def ci(key: str) -> dict:
        return mean_ci95([float(row[key]) for row in group])

    def frac(key: str) -> float:
        return float(np.mean([bool(row[key]) for row in group]))

    return {
        "matched": {
            "discovery_advantage": ci("matched_discovery_advantage"),
            "compute_opportunity_gain": ci("matched_compute_opportunity_gain"),
            "equal_compute_gap_wide_minus_compact": ci(
                "matched_equal_compute_gap_wide_minus_compact"
            ),
            "break_even_margin": ci("matched_break_even_margin"),
            "wide_win_fraction_same_steps": frac("wide_beats_matched_same_steps"),
            "wide_win_fraction_equal_compute": frac("wide_beats_matched_equal_compute"),
        },
        "native": {
            "discovery_advantage": ci("native_discovery_advantage"),
            "compute_opportunity_gain": ci("native_compute_opportunity_gain"),
            "equal_compute_gap_wide_minus_compact": ci(
                "native_equal_compute_gap_wide_minus_compact"
            ),
            "break_even_margin": ci("native_break_even_margin"),
            "wide_win_fraction_same_steps": frac("wide_beats_native_same_steps"),
            "wide_win_fraction_equal_compute": frac("wide_beats_native_equal_compute"),
        },
        "structural_damage_vs_dense": ci("same_step_structural_damage_vs_dense"),
        "singular_value_tail_fraction": ci("singular_value_tail_fraction"),
        "mean_compact_extra_steps": float(np.mean([row["compact_extra_steps"] for row in group])),
        "wide_to_compact_cost_ratio": float(np.mean([row["wide_to_compact_cost_ratio"] for row in group])),
        "parameter_fraction_removed": float(np.mean([row["parameter_fraction_removed"] for row in group])),
        "max_abs_break_even_identity_error": float(
            max(
                max(abs(row["matched_identity_error"]), abs(row["native_identity_error"]))
                for row in group
            )
        ),
    }


def summarize(results: list[dict]) -> dict:
    rows = [row for result in results for row in result["records"]]
    ranks = sorted({int(row["target_rank"]) for row in rows})
    checkpoints = sorted({int(row["checkpoint"]) for row in rows})
    by_rank: dict[str, dict] = {}

    for rank in ranks:
        rank_rows = [row for row in rows if int(row["target_rank"]) == rank]
        by_checkpoint = {
            str(checkpoint): _checkpoint_summary(
                [row for row in rank_rows if int(row["checkpoint"]) == checkpoint]
            )
            for checkpoint in checkpoints
        }
        post = [row for row in rank_rows if int(row["checkpoint"]) > 0]
        by_rank[str(rank)] = {
            "by_checkpoint": by_checkpoint,
            "post_initialization_seed_aggregated": {
                "matched_discovery_advantage": _seed_aggregated_ci(
                    post, "matched_discovery_advantage"
                ),
                "matched_compute_opportunity_gain": _seed_aggregated_ci(
                    post, "matched_compute_opportunity_gain"
                ),
                "matched_equal_compute_gap_wide_minus_compact": _seed_aggregated_ci(
                    post, "matched_equal_compute_gap_wide_minus_compact"
                ),
                "matched_break_even_margin": _seed_aggregated_ci(
                    post, "matched_break_even_margin"
                ),
                "native_discovery_advantage": _seed_aggregated_ci(
                    post, "native_discovery_advantage"
                ),
                "native_compute_opportunity_gain": _seed_aggregated_ci(
                    post, "native_compute_opportunity_gain"
                ),
                "native_equal_compute_gap_wide_minus_compact": _seed_aggregated_ci(
                    post, "native_equal_compute_gap_wide_minus_compact"
                ),
                "native_break_even_margin": _seed_aggregated_ci(
                    post, "native_break_even_margin"
                ),
                "matched_equal_compute_win_fraction_all_rows": float(
                    np.mean([row["wide_beats_matched_equal_compute"] for row in post])
                ),
                "native_equal_compute_win_fraction_all_rows": float(
                    np.mean([row["wide_beats_native_equal_compute"] for row in post])
                ),
            },
        }

    return {
        "by_rank": by_rank,
        "note": (
            "Checkpoint CIs use paired independent seeds (n = number of seeds). Overall post-initialization "
            "CIs first average repeated checkpoints within each seed and then compute the CI across seed means; "
            "they do not treat checkpoint observations from the same seed as independent."
        ),
    }


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
    parser.add_argument(
        "--checkpoints",
        type=int,
        nargs="+",
        default=[0, 20, 40, 60, 80, 100, 120],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/svd_break_even_sweep.json"),
    )
    args = parser.parse_args()

    if args.teacher_rank >= args.student_rank:
        raise ValueError("teacher-rank must be smaller than student-rank")
    if any(rank < 1 or rank >= args.student_rank for rank in args.target_ranks):
        raise ValueError("target ranks must lie between 1 and student_rank - 1")
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
        )
        for seed in args.seeds
    ]

    payload = {
        "status": "temporary_width_break_even_decomposition",
        "warning": (
            "Compute matching uses an analytic factorized-matmul MAC proxy, not measured hardware FLOPs. "
            "The matched compact control is the target-free SVD recompression of the identical initial wide "
            "model; the native compact control is an independently initialized rank-r model."
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
