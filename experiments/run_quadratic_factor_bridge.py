from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from scipy.stats import t as student_t

from neural_net.quadratic_factor_net import (
    BilinearQuadraticFactor,
    factor_forward_macs,
    population_matrix_risk,
)


def teacher(dimension: int, rank: int, theta: float, dtype: torch.dtype):
    basis = torch.eye(dimension, dtype=dtype)[:, :rank]
    a_star = theta * (basis @ basis.T)
    return basis, a_star


def matched_initialization(
    seed: int,
    dimension: int,
    wide_factors: int,
    dtype: torch.dtype,
):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    weight = torch.randn(
        dimension,
        wide_factors,
        generator=generator,
        dtype=dtype,
    ) / math.sqrt(wide_factors)
    return weight, generator


def data_stream(
    generator: torch.Generator,
    steps: int,
    batch_size: int,
    dimension: int,
    dtype: torch.dtype,
    a_star: torch.Tensor,
    noise_std: float,
):
    x = torch.randn(
        steps,
        batch_size,
        dimension,
        generator=generator,
        dtype=dtype,
    )
    z = torch.randn(
        steps,
        batch_size,
        dimension,
        generator=generator,
        dtype=dtype,
    )
    y = torch.einsum("sbi,ij,sbj->sb", x, a_star, z)
    if noise_std > 0.0:
        y = y + noise_std * torch.randn(
            steps,
            batch_size,
            generator=generator,
            dtype=dtype,
        )
    return x, z, y


def sgd_steps(
    model: BilinearQuadraticFactor,
    x: torch.Tensor,
    z: torch.Tensor,
    y: torch.Tensor,
    start: int,
    stop: int,
    learning_rate: float,
) -> None:
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    for step in range(int(start), int(stop)):
        prediction = model(x[step], z[step])
        loss = 0.5 * torch.mean((prediction - y[step]) ** 2)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


def spectral_state(
    model: BilinearQuadraticFactor,
    teacher_basis: torch.Tensor,
    retained_rank: int,
) -> dict:
    with torch.no_grad():
        covariance = model.covariance().double()
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        top = eigenvectors[:, -retained_rank:]
        overlap = float(
            torch.sum((top.T @ teacher_basis.double()) ** 2).detach().cpu()
            / retained_rank
        )
        signal_floor = float(eigenvalues[-retained_rank].detach().cpu())
        bulk_edge = (
            float(eigenvalues[-retained_rank - 1].detach().cpu())
            if retained_rank < len(eigenvalues)
            else 0.0
        )
        eigengap = signal_floor - bulk_edge
        if retained_rank < len(eigenvalues):
            positive = eigenvalues.clamp_min(0.0)
            total = float(positive.sum().detach().cpu())
            bulk_trace_fraction = (
                float(positive[:-retained_rank].sum().detach().cpu()) / total
                if total > 0.0
                else 0.0
            )
        else:
            bulk_trace_fraction = 0.0
    return {
        "teacher_subspace_overlap": overlap,
        "signal_floor": signal_floor,
        "bulk_edge": bulk_edge,
        "eigengap": eigengap,
        "bulk_trace_fraction": bulk_trace_fraction,
    }


def mean_simultaneous_ci(
    values,
    *,
    alpha: float,
    simultaneous_tests: int,
) -> dict:
    x = np.asarray(list(values), dtype=float)
    n = len(x)
    mean = float(x.mean())
    if n <= 1:
        return {
            "n": n,
            "mean": mean,
            "std": 0.0,
            "se": 0.0,
            "ci_lo": mean,
            "ci_hi": mean,
        }
    std = float(x.std(ddof=1))
    se = std / math.sqrt(n)
    critical = float(
        student_t.ppf(
            1.0 - alpha / (2.0 * simultaneous_tests),
            df=n - 1,
        )
    )
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "se": float(se),
        "ci_lo": float(mean - critical * se),
        "ci_hi": float(mean + critical * se),
    }


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
    dtype: torch.dtype,
) -> dict:
    teacher_basis, a_star = teacher(dimension, teacher_rank, theta, dtype)
    initial_weight, generator = matched_initialization(
        seed,
        dimension,
        wide_factors,
        dtype,
    )
    wide = BilinearQuadraticFactor.from_weight(initial_weight)
    compact_initial = wide.svd_contracted_copy(teacher_rank)

    wide_macs = factor_forward_macs(dimension, wide_factors)
    compact_macs = factor_forward_macs(dimension, teacher_rank)
    cost_ratio = wide_macs / compact_macs

    equal_compute_steps = {
        int(checkpoint): int(math.ceil(horizon_steps + cost_ratio * checkpoint))
        for checkpoint in checkpoints
    }
    max_equal = max(equal_compute_steps.values())
    max_wide = max(checkpoints) + horizon_steps
    max_steps = max(max_equal, max_wide)
    x, z, y = data_stream(
        generator,
        max_steps,
        batch_size,
        dimension,
        dtype,
        a_star,
        noise_std,
    )

    # Train the matched compact path once and cache every budget we will query.
    compact = compact_initial.clone()
    compact_risks = {0: population_matrix_risk(compact, a_star)}
    compact_needed = sorted(
        set(
            [int(checkpoint + horizon_steps) for checkpoint in checkpoints]
            + list(equal_compute_steps.values())
        )
    )
    last = 0
    for stop in compact_needed:
        sgd_steps(compact, x, z, y, last, stop, learning_rate)
        last = stop
        compact_risks[stop] = population_matrix_risk(compact, a_star)

    # Train the wide path once. Future dense controls are simply later points on
    # this same matched trajectory, which avoids redundant retraining noise.
    wide_needed = sorted(
        set(
            [int(checkpoint) for checkpoint in checkpoints]
            + [int(checkpoint + horizon_steps) for checkpoint in checkpoints]
        )
    )
    wide_risks: dict[int, float] = {}
    wide_weights: dict[int, torch.Tensor] = {}
    wide_spectral_states: dict[int, dict] = {}
    last = 0
    for stop in wide_needed:
        sgd_steps(wide, x, z, y, last, stop, learning_rate)
        last = stop
        wide_risks[stop] = population_matrix_risk(wide, a_star)
        if stop in checkpoints:
            wide_weights[stop] = wide.weight.detach().clone()
            wide_spectral_states[stop] = spectral_state(
                wide,
                teacher_basis,
                teacher_rank,
            )

    records = []
    for checkpoint in checkpoints:
        checkpoint = int(checkpoint)
        current = BilinearQuadraticFactor.from_weight(wide_weights[checkpoint])
        reduced = current.svd_contracted_copy(teacher_rank)
        wide_current_risk = wide_risks[checkpoint]
        reduced_current_risk = population_matrix_risk(reduced, a_star)

        sgd_steps(
            reduced,
            x,
            z,
            y,
            checkpoint,
            checkpoint + horizon_steps,
            learning_rate,
        )
        reduced_future_risk = population_matrix_risk(reduced, a_star)
        dense_future_risk = wide_risks[checkpoint + horizon_steps]

        same_steps = checkpoint + horizon_steps
        equal_steps = equal_compute_steps[checkpoint]
        compact_same_risk = compact_risks[same_steps]
        compact_equal_risk = compact_risks[equal_steps]

        discovery_advantage = compact_same_risk - reduced_future_risk
        compute_opportunity_gain = compact_same_risk - compact_equal_risk
        equal_compute_margin = compact_equal_risk - reduced_future_risk

        records.append(
            {
                "seed": int(seed),
                "checkpoint": checkpoint,
                "horizon_steps": int(horizon_steps),
                "wide_forward_macs": int(wide_macs),
                "compact_forward_macs": int(compact_macs),
                "wide_to_compact_mac_ratio": float(cost_ratio),
                "wide_current_population_risk": float(wide_current_risk),
                "contracted_current_population_risk": float(reduced_current_risk),
                "immediate_contraction_jump": float(
                    reduced_current_risk - wide_current_risk
                ),
                "wide_future_population_risk": float(dense_future_risk),
                "contracted_future_population_risk": float(reduced_future_risk),
                "future_structural_damage": float(
                    reduced_future_risk - dense_future_risk
                ),
                "compact_same_steps_population_risk": float(compact_same_risk),
                "compact_equal_compute_population_risk": float(compact_equal_risk),
                "compact_equal_compute_steps": int(equal_steps),
                "discovery_advantage_D": float(discovery_advantage),
                "compute_opportunity_gain_O": float(compute_opportunity_gain),
                "equal_compute_margin_D_minus_O": float(equal_compute_margin),
                "wide_wins_equal_compute": bool(equal_compute_margin > 0.0),
                **wide_spectral_states[checkpoint],
            }
        )

    return {
        "seed": int(seed),
        "wide_to_compact_mac_ratio": float(cost_ratio),
        "records": records,
    }


def summarize(
    runs: list[dict],
    checkpoints: list[int],
    alpha: float,
) -> dict:
    rows = [record for run in runs for record in run["records"]]
    simultaneous_tests = len(checkpoints)
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
            "equal_compute_margin_D_minus_O": mean_simultaneous_ci(
                (row["equal_compute_margin_D_minus_O"] for row in group),
                alpha=alpha,
                simultaneous_tests=simultaneous_tests,
            ),
            "wide_win_fraction": float(
                np.mean([row["wide_wins_equal_compute"] for row in group])
            ),
        }

    safe = [
        int(checkpoint)
        for checkpoint in checkpoints
        if by_checkpoint[str(checkpoint)]["future_structural_damage"]["ci_hi"] <= 0.0
    ]
    discovery_positive = [
        int(checkpoint)
        for checkpoint in checkpoints
        if by_checkpoint[str(checkpoint)]["discovery_advantage_D"]["ci_lo"] > 0.0
    ]
    equal_compute_positive = [
        int(checkpoint)
        for checkpoint in checkpoints
        if by_checkpoint[str(checkpoint)]["equal_compute_margin_D_minus_O"]["ci_lo"]
        > 0.0
    ]
    best = max(
        checkpoints,
        key=lambda checkpoint: by_checkpoint[str(checkpoint)][
            "equal_compute_margin_D_minus_O"
        ]["mean"],
    )

    return {
        "simultaneous_ci_family_alpha": float(alpha),
        "simultaneous_checkpoint_count": int(simultaneous_tests),
        "by_checkpoint": by_checkpoint,
        "earliest_simultaneously_safe_checkpoint": safe[0] if safe else None,
        "earliest_simultaneously_positive_discovery_checkpoint": (
            discovery_positive[0] if discovery_positive else None
        ),
        "earliest_simultaneously_positive_equal_compute_checkpoint": (
            equal_compute_positive[0] if equal_compute_positive else None
        ),
        "best_mean_equal_compute_checkpoint": int(best),
        "mechanism_pass": bool(safe and discovery_positive),
        "equal_compute_pass": bool(equal_compute_positive),
        "interpretation": (
            "D>0 means temporary wide training creates a better compact state at matched update count. "
            "S<=0 means contraction does not hurt the matched future wide trajectory. M=D-O>0 means "
            "the discovery advantage also repays the architecture-implied wide/compact forward-MAC ratio."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(100, 150)))
    parser.add_argument("--dimension", type=int, default=32)
    parser.add_argument("--wide-factors", type=int, default=48)
    parser.add_argument("--teacher-rank", type=int, default=12)
    parser.add_argument("--theta", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--horizon-steps", type=int, default=40)
    parser.add_argument(
        "--checkpoints",
        type=int,
        nargs="+",
        default=[20, 40, 60, 80, 100, 120, 160],
    )
    parser.add_argument("--noise-std", type=float, default=0.0)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/quadratic_factor_bridge.json"),
    )
    args = parser.parse_args()

    if not 1 <= args.teacher_rank < args.wide_factors:
        raise ValueError("teacher rank must lie in [1, wide_factors)")
    if args.teacher_rank > args.dimension:
        raise ValueError("teacher rank cannot exceed ambient dimension")
    if args.wide_factors < args.dimension:
        raise ValueError(
            "the confirmatory bridge requires wide_factors >= dimension so the initial covariance can be full rank"
        )
    if args.theta <= 0.0 or args.learning_rate <= 0.0:
        raise ValueError("theta and learning rate must be positive")
    if args.batch_size < 1 or args.horizon_steps < 1:
        raise ValueError("batch size and horizon must be positive")
    if any(checkpoint <= 0 for checkpoint in args.checkpoints):
        raise ValueError("all checkpoints must be positive")
    if not 0.0 < args.alpha < 1.0:
        raise ValueError("alpha must lie in (0,1)")

    torch.set_num_threads(1)
    dtype = torch.float32
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
            dtype=dtype,
        )
        for seed in args.seeds
    ]

    payload = {
        "status": "precommitted_quadratic_factor_bridge",
        "warning": (
            "Seeds 0--9 are development-only. Default seeds 100--149 are confirmatory. "
            "SVD contraction is target-free; the teacher subspace is used only for evaluation. "
            "Equal-compute accounting uses analytic factor-forward MACs and is not a hardware wall-clock claim."
        ),
        "config": vars(args) | {"output": str(args.output), "dtype": "float32"},
        "runs": runs,
        "summary": summarize(runs, args.checkpoints, args.alpha),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
