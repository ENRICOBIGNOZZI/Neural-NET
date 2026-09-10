from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from neural_net.quadratic_factor_net import (
    BilinearQuadraticFactor,
    population_matrix_risk,
)
from run_quadratic_factor_bridge import (
    data_stream,
    matched_initialization,
    sgd_steps,
    teacher,
)


def target_free_spectral_state(model: BilinearQuadraticFactor) -> dict:
    """Return rank/onset diagnostics using only the current covariance spectrum."""
    with torch.no_grad():
        vals = torch.linalg.eigvalsh(model.covariance().double()).cpu().numpy()[::-1]
    vals = np.maximum(vals, 0.0)
    d = len(vals)
    if d < 3:
        raise ValueError("development detector requires dimension >= 3")

    scale = max(float(vals[0]), 1.0)
    eps = 1e-12 * scale
    gaps = vals[:-1] - vals[1:]
    order = np.argsort(gaps)[::-1]
    best = int(order[0])
    second_gap = float(gaps[order[1]]) if len(order) > 1 else 0.0
    largest_rank = best + 1
    largest_gap = float(gaps[best])
    gap_dominance = largest_gap / max(second_gap, eps)
    boundary_ratio = float(vals[best] / max(vals[best + 1], eps))

    # Independent detector: best two-cluster split of the log spectrum.
    # No teacher information, target labels, or candidate target rank enter here.
    logvals = np.log(vals + eps)
    split_scores = []
    for k in range(1, d):
        top = logvals[:k]
        bottom = logvals[k:]
        sse = float(np.sum((top - top.mean()) ** 2) + np.sum((bottom - bottom.mean()) ** 2))
        split_scores.append(sse)
    logsplit_rank = int(np.argmin(split_scores)) + 1
    k = logsplit_rank
    top = logvals[:k]
    bottom = logvals[k:]
    pooled = math.sqrt(float(np.var(top) + np.var(bottom)) + 1e-12)
    logsplit_separation = float((top.mean() - bottom.mean()) / pooled)
    logsplit_boundary_ratio = float(vals[k - 1] / max(vals[k], eps))

    total = float(vals.sum())
    largest_top_mass_fraction = float(vals[:largest_rank].sum() / total) if total > 0 else 0.0
    logsplit_top_mass_fraction = float(vals[:logsplit_rank].sum() / total) if total > 0 else 0.0

    return {
        "eigenvalues_desc": [float(x) for x in vals],
        "largest_gap_rank": largest_rank,
        "largest_gap": largest_gap,
        "second_largest_gap": second_gap,
        "gap_dominance": float(gap_dominance),
        "largest_gap_boundary_ratio": boundary_ratio,
        "largest_gap_top_mass_fraction": largest_top_mass_fraction,
        "logsplit_rank": logsplit_rank,
        "logsplit_sse": float(split_scores[k - 1]),
        "logsplit_separation": logsplit_separation,
        "logsplit_boundary_ratio": logsplit_boundary_ratio,
        "logsplit_top_mass_fraction": logsplit_top_mass_fraction,
    }


def train_contracted_future(
    weight: torch.Tensor,
    factors: int,
    x: torch.Tensor,
    z: torch.Tensor,
    y: torch.Tensor,
    checkpoint: int,
    horizon_steps: int,
    learning_rate: float,
    a_star: torch.Tensor,
) -> float:
    factors = int(factors)
    if not 1 <= factors <= min(weight.shape):
        return math.inf
    reduced = BilinearQuadraticFactor.from_weight(weight).svd_contracted_copy(factors)
    sgd_steps(
        reduced,
        x,
        z,
        y,
        checkpoint,
        checkpoint + horizon_steps,
        learning_rate,
    )
    return population_matrix_risk(reduced, a_star)


def run_seed(
    seed: int,
    *,
    dimension: int,
    wide_factors: int,
    teacher_rank: int,
    theta: float,
    learning_rate: float,
    batch_size: int,
    horizon_steps: int,
    checkpoints: list[int],
    noise_std: float,
) -> dict:
    dtype = torch.float32
    teacher_basis, a_star = teacher(dimension, teacher_rank, theta, dtype)
    initial_weight, generator = matched_initialization(seed, dimension, wide_factors, dtype)
    wide = BilinearQuadraticFactor.from_weight(initial_weight)

    max_steps = max(checkpoints) + horizon_steps
    x, z, y = data_stream(
        generator,
        max_steps,
        batch_size,
        dimension,
        dtype,
        a_star,
        noise_std,
    )

    needed = sorted(set(checkpoints + [c + horizon_steps for c in checkpoints]))
    wide_risks: dict[int, float] = {}
    weights: dict[int, torch.Tensor] = {}
    states: dict[int, dict] = {}
    last = 0
    for stop in needed:
        sgd_steps(wide, x, z, y, last, stop, learning_rate)
        last = stop
        wide_risks[stop] = population_matrix_risk(wide, a_star)
        if stop in checkpoints:
            weights[stop] = wide.weight.detach().clone()
            states[stop] = target_free_spectral_state(wide)

    records = []
    previous_largest_rank = None
    previous_logsplit_rank = None
    for checkpoint in checkpoints:
        state = states[checkpoint]
        dense_future = wide_risks[checkpoint + horizon_steps]

        largest_rank = int(state["largest_gap_rank"])
        logsplit_rank = int(state["logsplit_rank"])
        largest_future = train_contracted_future(
            weights[checkpoint],
            largest_rank,
            x,
            z,
            y,
            checkpoint,
            horizon_steps,
            learning_rate,
            a_star,
        )
        logsplit_future = train_contracted_future(
            weights[checkpoint],
            logsplit_rank,
            x,
            z,
            y,
            checkpoint,
            horizon_steps,
            learning_rate,
            a_star,
        )
        oracle_future = train_contracted_future(
            weights[checkpoint],
            teacher_rank,
            x,
            z,
            y,
            checkpoint,
            horizon_steps,
            learning_rate,
            a_star,
        )

        with torch.no_grad():
            covariance = BilinearQuadraticFactor.from_weight(weights[checkpoint]).covariance().double()
            _, vecs = torch.linalg.eigh(covariance)
            top = vecs[:, -teacher_rank:]
            oracle_overlap = float(
                torch.sum((top.T @ teacher_basis.double()) ** 2).cpu() / teacher_rank
            )

        records.append(
            {
                "seed": int(seed),
                "checkpoint": int(checkpoint),
                **state,
                "largest_rank_persistent": bool(previous_largest_rank == largest_rank),
                "logsplit_rank_persistent": bool(previous_logsplit_rank == logsplit_rank),
                "largest_rank_equals_teacher": bool(largest_rank == teacher_rank),
                "logsplit_rank_equals_teacher": bool(logsplit_rank == teacher_rank),
                "oracle_teacher_subspace_overlap": oracle_overlap,
                "oracle_future_structural_damage": float(oracle_future - dense_future),
                "largest_gap_future_structural_damage": float(largest_future - dense_future),
                "logsplit_future_structural_damage": float(logsplit_future - dense_future),
            }
        )
        previous_largest_rank = largest_rank
        previous_logsplit_rank = logsplit_rank

    return {"seed": int(seed), "records": records}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(50)))
    parser.add_argument("--dimension", type=int, default=24)
    parser.add_argument("--wide-factors", type=int, required=True)
    parser.add_argument("--teacher-rank", type=int, default=12)
    parser.add_argument("--theta", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--horizon-steps", type=int, default=40)
    parser.add_argument("--checkpoints", type=int, nargs="+", default=[20, 40, 60, 80, 100, 120])
    parser.add_argument("--noise-std", type=float, default=0.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.dimension != 24 or args.teacher_rank != 12:
        raise ValueError("development family freezes dimension=24 and teacher_rank=12 for evaluation")
    if args.wide_factors not in {24, 30, 36, 42, 48}:
        raise ValueError("wide_factors must be one of 24,30,36,42,48")
    if sorted(args.checkpoints) != [20, 40, 60, 80, 100, 120]:
        raise ValueError("development checkpoint grid is frozen")
    if set(args.seeds) != set(range(50)) or len(args.seeds) != 50:
        raise ValueError("development seed family is frozen to 0--49")

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
        )
        for seed in args.seeds
    ]

    payload = {
        "status": "nonoracle_spectral_onset_development",
        "warning": (
            "Teacher rank and target overlap are evaluation-only. The two reported rank detectors use only "
            "the current covariance eigenvalues. This development family may be used to freeze a controller; "
            "any controller claim must then be tested on disjoint confirmatory seeds."
        ),
        "config": vars(args) | {"output": str(args.output)},
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps({"wide_factors": args.wide_factors, "runs": len(runs)}, indent=2))


if __name__ == "__main__":
    main()
