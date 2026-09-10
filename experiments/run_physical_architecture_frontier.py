from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from run_quadratic_factor_bridge import mean_simultaneous_ci, run_seed


def summarize_architecture(
    runs: list[dict],
    checkpoints: list[int],
    *,
    alpha: float,
    simultaneous_tests: int,
) -> dict:
    rows = [record for run in runs for record in run["records"]]
    by_checkpoint = {}
    for checkpoint in checkpoints:
        group = [row for row in rows if int(row["checkpoint"]) == int(checkpoint)]
        by_checkpoint[str(checkpoint)] = {
            "teacher_subspace_overlap": mean_simultaneous_ci(
                (row["teacher_subspace_overlap"] for row in group),
                alpha=alpha,
                simultaneous_tests=simultaneous_tests,
            ),
            "eigengap": mean_simultaneous_ci(
                (row["eigengap"] for row in group),
                alpha=alpha,
                simultaneous_tests=simultaneous_tests,
            ),
            "bulk_trace_fraction": mean_simultaneous_ci(
                (row["bulk_trace_fraction"] for row in group),
                alpha=alpha,
                simultaneous_tests=simultaneous_tests,
            ),
            "future_structural_damage": mean_simultaneous_ci(
                (row["future_structural_damage"] for row in group),
                alpha=alpha,
                simultaneous_tests=simultaneous_tests,
            ),
            "discovery_advantage_D": mean_simultaneous_ci(
                (row["discovery_advantage_D"] for row in group),
                alpha=alpha,
                simultaneous_tests=simultaneous_tests,
            ),
            "compute_opportunity_gain_O": mean_simultaneous_ci(
                (row["compute_opportunity_gain_O"] for row in group),
                alpha=alpha,
                simultaneous_tests=simultaneous_tests,
            ),
            "equal_compute_margin_M": mean_simultaneous_ci(
                (row["equal_compute_margin_D_minus_O"] for row in group),
                alpha=alpha,
                simultaneous_tests=simultaneous_tests,
            ),
            "wide_win_fraction": float(
                np.mean([row["wide_wins_equal_compute"] for row in group])
            ),
        }

    safe = [
        int(c)
        for c in checkpoints
        if by_checkpoint[str(c)]["future_structural_damage"]["ci_hi"] <= 0.0
    ]
    discovery = [
        int(c)
        for c in checkpoints
        if by_checkpoint[str(c)]["discovery_advantage_D"]["ci_lo"] > 0.0
    ]
    equal_compute = [
        int(c)
        for c in checkpoints
        if by_checkpoint[str(c)]["equal_compute_margin_M"]["ci_lo"] > 0.0
    ]
    best = max(
        checkpoints,
        key=lambda c: by_checkpoint[str(c)]["equal_compute_margin_M"]["mean"],
    )
    return {
        "simultaneous_family_alpha": float(alpha),
        "simultaneous_total_cells": int(simultaneous_tests),
        "by_checkpoint": by_checkpoint,
        "earliest_safe_checkpoint": safe[0] if safe else None,
        "earliest_positive_discovery_checkpoint": discovery[0] if discovery else None,
        "earliest_positive_equal_compute_checkpoint": (
            equal_compute[0] if equal_compute else None
        ),
        "best_mean_equal_compute_checkpoint": int(best),
        "mechanism_positive": bool(discovery),
        "equal_compute_positive": bool(equal_compute),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(200, 250)))
    parser.add_argument("--dimension", type=int, default=24)
    parser.add_argument("--wide-factors", type=int, required=True)
    parser.add_argument("--teacher-rank", type=int, default=12)
    parser.add_argument("--theta", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--horizon-steps", type=int, default=40)
    parser.add_argument(
        "--checkpoints",
        type=int,
        nargs="+",
        default=[20, 40, 60, 80, 100, 120],
    )
    parser.add_argument("--noise-std", type=float, default=0.0)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--simultaneous-tests", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.wide_factors not in {24, 30, 36, 42, 48}:
        raise ValueError("confirmatory wide_factors must be one of 24,30,36,42,48")
    if args.dimension != 24 or args.teacher_rank != 12:
        raise ValueError("confirmatory frontier freezes dimension=24 and teacher_rank=12")
    if args.simultaneous_tests != 30:
        raise ValueError("confirmatory family is frozen to 30 architecture/checkpoint cells")
    if sorted(args.checkpoints) != [20, 40, 60, 80, 100, 120]:
        raise ValueError("confirmatory checkpoint grid is frozen")
    if set(args.seeds) != set(range(200, 250)) or len(args.seeds) != 50:
        raise ValueError("confirmatory seed family is frozen to 200--249")

    import torch

    torch.set_num_threads(1)
    runs = [
        run_seed(
            seed,
            dimension=args.dimension,
            wide_factors=args.wide_factors,
            teacher_rank=args.teacher_rank,
            theta=args.theta,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            horizon_steps=args.horizon_steps,
            checkpoints=args.checkpoints,
            noise_std=args.noise_std,
            dtype=torch.float32,
        )
        for seed in args.seeds
    ]
    ratio = runs[0]["wide_to_compact_mac_ratio"]
    payload = {
        "status": "precommitted_physical_architecture_frontier",
        "warning": (
            "Seeds 200--249 are confirmatory and disjoint from prior confirmatory samples. "
            "Changing wide factor count changes both architecture-implied compute ratio and Wishart aspect ratio; "
            "this is a physical architecture frontier, not a pure cost-ratio intervention."
        ),
        "config": vars(args) | {"output": str(args.output), "dtype": "float32"},
        "wide_to_compact_mac_ratio": float(ratio),
        "runs": runs,
        "summary": summarize_architecture(
            runs,
            args.checkpoints,
            alpha=args.alpha,
            simultaneous_tests=args.simultaneous_tests,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps({"wide_factors": args.wide_factors, "ratio": ratio, **payload["summary"]}, indent=2))


if __name__ == "__main__":
    main()
