from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from neural_net.quadratic_factor_net import (
    BilinearQuadraticFactor,
    factor_forward_macs,
    population_matrix_risk,
)
from neural_net.spectral_phase_controller import (
    DualSpectrumPhaseController,
    SpectralPhaseConfig,
)
from run_quadratic_factor_bridge import (
    data_stream,
    matched_initialization,
    sgd_steps,
    teacher,
)


def covariance_eigenvalues_desc(weight: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        covariance = BilinearQuadraticFactor.from_weight(weight).covariance().double()
        return torch.linalg.eigvalsh(covariance).cpu().numpy()[::-1]


def teacher_support_coverage(
    weight: torch.Tensor,
    retained_rank: int,
    teacher_basis: torch.Tensor,
) -> float:
    with torch.no_grad():
        covariance = BilinearQuadraticFactor.from_weight(weight).covariance().double()
        _, vecs = torch.linalg.eigh(covariance)
        top = vecs[:, -int(retained_rank):]
        return float(
            torch.sum((top.T @ teacher_basis.double()) ** 2).cpu()
            / teacher_basis.shape[1]
        )


def contracted_future_risk(
    weight: torch.Tensor,
    retained_rank: int,
    x: torch.Tensor,
    z: torch.Tensor,
    y: torch.Tensor,
    checkpoint: int,
    horizon_steps: int,
    learning_rate: float,
    a_star: torch.Tensor,
) -> float:
    reduced = BilinearQuadraticFactor.from_weight(weight).svd_contracted_copy(
        int(retained_rank)
    )
    sgd_steps(
        reduced,
        x,
        z,
        y,
        int(checkpoint),
        int(checkpoint + horizon_steps),
        learning_rate,
    )
    return population_matrix_risk(reduced, a_star)


def append_stream(
    generator: torch.Generator,
    x: torch.Tensor,
    z: torch.Tensor,
    y: torch.Tensor,
    *,
    extra_steps: int,
    batch_size: int,
    dimension: int,
    dtype: torch.dtype,
    a_star: torch.Tensor,
    noise_std: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if extra_steps <= 0:
        return x, z, y
    x_extra, z_extra, y_extra = data_stream(
        generator,
        int(extra_steps),
        batch_size,
        dimension,
        dtype,
        a_star,
        noise_std,
    )
    return (
        torch.cat([x, x_extra], dim=0),
        torch.cat([z, z_extra], dim=0),
        torch.cat([y, y_extra], dim=0),
    )


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
    initial_weight, generator = matched_initialization(
        seed,
        dimension,
        wide_factors,
        dtype,
    )
    wide = BilinearQuadraticFactor.from_weight(initial_weight)
    controller = DualSpectrumPhaseController()

    max_wide_steps = max(checkpoints) + horizon_steps
    x, z, y = data_stream(
        generator,
        max_wide_steps,
        batch_size,
        dimension,
        dtype,
        a_star,
        noise_std,
    )

    needed = sorted(
        set(
            [int(c) for c in checkpoints]
            + [int(c + horizon_steps) for c in checkpoints]
        )
    )
    wide_risks: dict[int, float] = {}
    wide_weights: dict[int, torch.Tensor] = {}
    decisions: dict[int, object] = {}
    last = 0
    for stop in needed:
        sgd_steps(wide, x, z, y, last, stop, learning_rate)
        last = stop
        wide_risks[stop] = population_matrix_risk(wide, a_star)
        if stop in checkpoints:
            weight = wide.weight.detach().clone()
            wide_weights[stop] = weight
            decisions[stop] = controller.evaluate_eigenvalues(
                covariance_eigenvalues_desc(weight)
            )

    # Evaluation-only oracle: when would a contraction to the true teacher rank
    # become safe? This quantity never enters the controller.
    teacher_damage_by_checkpoint: dict[int, float] = {}
    for checkpoint in checkpoints:
        future = contracted_future_risk(
            wide_weights[checkpoint],
            teacher_rank,
            x,
            z,
            y,
            checkpoint,
            horizon_steps,
            learning_rate,
            a_star,
        )
        teacher_damage_by_checkpoint[int(checkpoint)] = float(
            future - wide_risks[checkpoint + horizon_steps]
        )
    teacher_safe = [
        int(c)
        for c in checkpoints
        if teacher_damage_by_checkpoint[int(c)] <= 0.0
    ]
    teacher_oracle_safe_checkpoint = teacher_safe[0] if teacher_safe else None

    trigger_checkpoint = None
    trigger_decision = None
    for checkpoint in checkpoints:
        decision = decisions[int(checkpoint)]
        if decision.should_contract:
            trigger_checkpoint = int(checkpoint)
            trigger_decision = decision
            break

    checkpoint_diagnostics = []
    for checkpoint in checkpoints:
        decision = decisions[int(checkpoint)]
        checkpoint_diagnostics.append(
            {
                "checkpoint": int(checkpoint),
                "controller_should_contract": bool(decision.should_contract),
                "controller_proposed_rank": int(decision.proposed_rank),
                "largest_gap_rank": int(decision.largest_gap_rank),
                "logsplit_rank": int(decision.logsplit_rank),
                "boundary_ratio": float(decision.boundary_ratio),
                "retained_mass_fraction": float(
                    decision.retained_mass_fraction
                ),
                "gap_dominance": float(decision.gap_dominance),
                "logsplit_separation": float(decision.logsplit_separation),
                "controller_reasons": list(decision.reasons),
                "evaluation_only_teacher_rank_damage": float(
                    teacher_damage_by_checkpoint[int(checkpoint)]
                ),
            }
        )

    if trigger_checkpoint is None or trigger_decision is None:
        return {
            "seed": int(seed),
            "triggered": False,
            "teacher_oracle_safe_checkpoint": teacher_oracle_safe_checkpoint,
            "missed_teacher_safe_opportunity": bool(
                teacher_oracle_safe_checkpoint is not None
            ),
            "checkpoint_diagnostics": checkpoint_diagnostics,
        }

    checkpoint = int(trigger_checkpoint)
    retained_rank = int(trigger_decision.proposed_rank)

    # Onset calibration is evaluated against the same structural action chosen
    # by the controller, not against a different oracle rank.
    selected_damage_by_checkpoint: dict[int, float] = {}
    for c in checkpoints:
        future = contracted_future_risk(
            wide_weights[c],
            retained_rank,
            x,
            z,
            y,
            c,
            horizon_steps,
            learning_rate,
            a_star,
        )
        selected_damage_by_checkpoint[int(c)] = float(
            future - wide_risks[c + horizon_steps]
        )
    selected_safe = [
        int(c)
        for c in checkpoints
        if selected_damage_by_checkpoint[int(c)] <= 0.0
    ]
    selected_oracle_safe_checkpoint = selected_safe[0] if selected_safe else None
    trigger_before_selected_oracle_safe = bool(
        selected_oracle_safe_checkpoint is None
        or checkpoint < selected_oracle_safe_checkpoint
    )

    reduced = BilinearQuadraticFactor.from_weight(
        wide_weights[checkpoint]
    ).svd_contracted_copy(retained_rank)
    reduced_current_risk = population_matrix_risk(reduced, a_star)
    wide_current_risk = wide_risks[checkpoint]
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
    structural_damage = reduced_future_risk - dense_future_risk

    wide_macs = factor_forward_macs(dimension, wide_factors)
    compact_macs = factor_forward_macs(dimension, retained_rank)
    cost_ratio = wide_macs / compact_macs
    same_steps = checkpoint + horizon_steps
    equal_steps = int(math.ceil(horizon_steps + cost_ratio * checkpoint))

    x_full, z_full, y_full = append_stream(
        generator,
        x,
        z,
        y,
        extra_steps=max(0, equal_steps - len(x)),
        batch_size=batch_size,
        dimension=dimension,
        dtype=dtype,
        a_star=a_star,
        noise_std=noise_std,
    )
    compact = BilinearQuadraticFactor.from_weight(
        initial_weight
    ).svd_contracted_copy(retained_rank)
    sgd_steps(compact, x_full, z_full, y_full, 0, same_steps, learning_rate)
    compact_same_risk = population_matrix_risk(compact, a_star)
    if equal_steps > same_steps:
        sgd_steps(
            compact,
            x_full,
            z_full,
            y_full,
            same_steps,
            equal_steps,
            learning_rate,
        )
    compact_equal_risk = population_matrix_risk(compact, a_star)

    discovery_advantage = compact_same_risk - reduced_future_risk
    compute_opportunity_gain = compact_same_risk - compact_equal_risk
    equal_compute_margin = compact_equal_risk - reduced_future_risk

    coverage = teacher_support_coverage(
        wide_weights[checkpoint],
        retained_rank,
        teacher_basis,
    )

    return {
        "seed": int(seed),
        "triggered": True,
        "trigger_checkpoint": checkpoint,
        "selected_rank": retained_rank,
        "rank_equals_teacher": bool(retained_rank == teacher_rank),
        "under_ranked_teacher": bool(retained_rank < teacher_rank),
        "over_ranked_teacher": bool(retained_rank > teacher_rank),
        "evaluation_only_teacher_support_coverage": float(coverage),
        "teacher_oracle_safe_checkpoint": teacher_oracle_safe_checkpoint,
        "selected_rank_oracle_safe_checkpoint": selected_oracle_safe_checkpoint,
        "trigger_before_selected_oracle_safe": trigger_before_selected_oracle_safe,
        "trigger_delay_from_selected_oracle_safe": (
            None
            if selected_oracle_safe_checkpoint is None
            else int(checkpoint - selected_oracle_safe_checkpoint)
        ),
        "wide_forward_macs": int(wide_macs),
        "selected_compact_forward_macs": int(compact_macs),
        "wide_to_selected_compact_mac_ratio": float(cost_ratio),
        "compact_equal_compute_steps": int(equal_steps),
        "wide_current_population_risk": float(wide_current_risk),
        "contracted_current_population_risk": float(reduced_current_risk),
        "immediate_contraction_jump": float(
            reduced_current_risk - wide_current_risk
        ),
        "wide_future_population_risk": float(dense_future_risk),
        "contracted_future_population_risk": float(reduced_future_risk),
        "future_structural_damage_S": float(structural_damage),
        "compact_same_steps_population_risk": float(compact_same_risk),
        "compact_equal_compute_population_risk": float(compact_equal_risk),
        "discovery_advantage_D": float(discovery_advantage),
        "compute_opportunity_gain_O": float(compute_opportunity_gain),
        "equal_compute_margin_M": float(equal_compute_margin),
        "wide_wins_equal_compute": bool(equal_compute_margin > 0.0),
        "controller_boundary_ratio": float(trigger_decision.boundary_ratio),
        "controller_retained_mass_fraction": float(
            trigger_decision.retained_mass_fraction
        ),
        "controller_gap_dominance": float(trigger_decision.gap_dominance),
        "controller_logsplit_separation": float(
            trigger_decision.logsplit_separation
        ),
        "selected_rank_damage_by_checkpoint": {
            str(k): float(v) for k, v in selected_damage_by_checkpoint.items()
        },
        "checkpoint_diagnostics": checkpoint_diagnostics,
    }


def summarize_runs(runs: list[dict]) -> dict:
    triggered = [run for run in runs if run["triggered"]]
    trigger_checkpoints = [run["trigger_checkpoint"] for run in triggered]
    return {
        "runs": len(runs),
        "triggered": len(triggered),
        "trigger_rate": float(len(triggered) / len(runs)),
        "exact_teacher_rank_triggers": int(
            sum(run["rank_equals_teacher"] for run in triggered)
        ),
        "under_rank_events": int(
            sum(run["under_ranked_teacher"] for run in triggered)
        ),
        "over_rank_events": int(
            sum(run["over_ranked_teacher"] for run in triggered)
        ),
        "trigger_before_selected_oracle_safe_events": int(
            sum(
                run["trigger_before_selected_oracle_safe"]
                for run in triggered
            )
        ),
        "missed_teacher_safe_opportunities": int(
            sum(
                run.get("missed_teacher_safe_opportunity", False)
                for run in runs
            )
        ),
        "median_trigger_checkpoint": (
            float(np.median(trigger_checkpoints))
            if trigger_checkpoints
            else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(300, 350)))
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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.dimension != 24 or args.teacher_rank != 12:
        raise ValueError(
            "confirmatory family freezes d=24 and teacher_rank=12 for evaluation"
        )
    if args.wide_factors not in {24, 30, 36, 42, 48}:
        raise ValueError("wide_factors must be one of 24,30,36,42,48")
    if sorted(args.checkpoints) != [20, 40, 60, 80, 100, 120]:
        raise ValueError("confirmatory checkpoint grid is frozen")
    if set(args.seeds) != set(range(300, 350)) or len(args.seeds) != 50:
        raise ValueError("confirmatory seeds are frozen to 300--349")

    cfg = SpectralPhaseConfig()
    if (
        cfg.min_boundary_ratio != 1.5
        or cfg.max_retained_mass_fraction != 0.90
    ):
        raise ValueError("controller thresholds differ from frozen protocol")

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
        "status": "precommitted_nonoracle_controller_confirmation",
        "warning": (
            "Controller selection uses current covariance eigenvalues only. "
            "Teacher rank, teacher support, oracle safe time, future structural "
            "damage, and equal-compute outcomes are evaluation-only."
        ),
        "controller": {
            "rank_rule": "largest_adjacent_gap_rank == best_two_cluster_log_spectrum_rank",
            "min_boundary_ratio": 1.5,
            "max_retained_mass_fraction": 0.90,
            "stopping_rule": "first frozen checkpoint satisfying all three gates",
        },
        "config": vars(args) | {"output": str(args.output), "dtype": "float32"},
        "runs": runs,
        "summary": summarize_runs(runs),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(
        json.dumps(
            {
                "wide_factors": args.wide_factors,
                **payload["summary"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
