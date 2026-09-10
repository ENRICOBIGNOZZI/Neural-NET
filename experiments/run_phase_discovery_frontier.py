from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from neural_net.quadratic_phase import (
    bbp_crossing_time,
    covariance_risk,
    exact_covariance_flow,
    null_bulk_edges,
)
from neural_net.rank_one_phase import (
    critical_wide_cost_ratio,
    rank_one_risk,
    rank_one_risk_after,
)


def mean_ci95(values) -> dict:
    x = np.asarray(list(values), dtype=float)
    n = len(x)
    mean = float(x.mean())
    if n <= 1:
        return {"n": n, "mean": mean, "std": 0.0, "se": 0.0, "ci95_lo": mean, "ci95_hi": mean}
    std = float(x.std(ddof=1))
    se = std / np.sqrt(n)
    radius = 1.96 * se
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "se": float(se),
        "ci95_lo": float(mean - radius),
        "ci95_hi": float(mean + radius),
    }


def wishart_initialization(seed: int, dimension: int, columns: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(dimension, columns))
    return (x @ x.T) / columns


def top_state(a: np.ndarray, teacher_direction: np.ndarray) -> tuple[float, float, float]:
    vals, vecs = np.linalg.eigh(a)
    top = float(vals[-1])
    second = float(vals[-2])
    v = vecs[:, -1]
    omega = float((teacher_direction @ v) ** 2)
    return top, omega, second


def run_seed(
    seed: int,
    *,
    dimension: int,
    columns: int,
    theta: float,
    compact_horizon: float,
    checkpoints: list[float],
    cost_ratios: list[float],
) -> dict:
    gamma = dimension / columns
    u = np.zeros(dimension)
    u[0] = 1.0
    a_star = theta * np.outer(u, u)
    a0 = wishart_initialization(seed, dimension, columns)

    compact_q0, compact_omega0, compact_second0 = top_state(a0, u)
    records = []
    for t in checkpoints:
        a_t = exact_covariance_flow(a0, a_star, t)
        top, omega, second = top_state(a_t, u)
        _, bulk_edge = null_bulk_edges(t, gamma)

        wide_full_risk = covariance_risk(a_t, a_star)
        contracted_risk_now = rank_one_risk(top, omega, theta)
        wide_then_compact_risk = rank_one_risk_after(top, omega, theta, compact_horizon)
        compact_same_time_risk = rank_one_risk_after(
            compact_q0,
            compact_omega0,
            theta,
            t + compact_horizon,
        )
        discovery_advantage = compact_same_time_risk - wide_then_compact_risk
        rho_star = critical_wide_cost_ratio(
            compact_q0,
            compact_omega0,
            top,
            omega,
            theta,
            t,
            compact_horizon,
        )

        ratio_results = {}
        for rho in cost_ratios:
            compact_equal_compute_risk = rank_one_risk_after(
                compact_q0,
                compact_omega0,
                theta,
                compact_horizon + rho * t,
            )
            opportunity_gain = compact_same_time_risk - compact_equal_compute_risk
            break_even_margin = compact_equal_compute_risk - wide_then_compact_risk
            ratio_results[str(rho)] = {
                "compact_equal_compute_risk": float(compact_equal_compute_risk),
                "compute_opportunity_gain": float(opportunity_gain),
                "break_even_margin_D_minus_O": float(break_even_margin),
                "wide_wins_equal_compute": bool(break_even_margin > 0.0),
            }

        records.append(
            {
                "seed": int(seed),
                "time": float(t),
                "theoretical_bbp_time": float(bbp_crossing_time(theta, gamma)),
                "pre_or_post_theoretical_bbp": "post" if t >= bbp_crossing_time(theta, gamma) else "pre",
                "theoretical_bulk_edge": float(bulk_edge),
                "top_eigenvalue": top,
                "second_eigenvalue": second,
                "empirical_top_gap": float(top - second),
                "top_minus_theoretical_bulk_edge": float(top - bulk_edge),
                "top_teacher_overlap": omega,
                "matched_compact_initial_top_eigenvalue": compact_q0,
                "matched_compact_initial_teacher_overlap": compact_omega0,
                "matched_compact_initial_second_eigenvalue": compact_second0,
                "wide_full_covariance_risk": float(wide_full_risk),
                "rank_one_contraction_jump": float(contracted_risk_now - wide_full_risk),
                "wide_then_compact_risk": float(wide_then_compact_risk),
                "matched_compact_same_time_risk": float(compact_same_time_risk),
                "discovery_advantage": float(discovery_advantage),
                "critical_wide_to_compact_cost_ratio": float(rho_star),
                "cost_ratio_results": ratio_results,
            }
        )

    return {
        "seed": int(seed),
        "gamma": float(gamma),
        "theoretical_bbp_time": float(bbp_crossing_time(theta, gamma)),
        "records": records,
    }


def summarize(runs: list[dict], cost_ratios: list[float]) -> dict:
    rows = [record for run in runs for record in run["records"]]
    checkpoints = sorted({float(row["time"]) for row in rows})
    by_time = {}
    for t in checkpoints:
        group = [row for row in rows if float(row["time"]) == t]
        rhos = np.asarray([row["critical_wide_to_compact_cost_ratio"] for row in group], dtype=float)
        finite_rhos = rhos[np.isfinite(rhos)]
        ratio_summary = {}
        for rho in cost_ratios:
            key = str(rho)
            margins = [row["cost_ratio_results"][key]["break_even_margin_D_minus_O"] for row in group]
            ratio_summary[key] = {
                "break_even_margin_D_minus_O": mean_ci95(margins),
                "wide_win_fraction": float(
                    np.mean([row["cost_ratio_results"][key]["wide_wins_equal_compute"] for row in group])
                ),
            }

        by_time[str(t)] = {
            "phase": group[0]["pre_or_post_theoretical_bbp"],
            "top_teacher_overlap": mean_ci95(row["top_teacher_overlap"] for row in group),
            "top_minus_theoretical_bulk_edge": mean_ci95(
                row["top_minus_theoretical_bulk_edge"] for row in group
            ),
            "empirical_top_gap": mean_ci95(row["empirical_top_gap"] for row in group),
            "discovery_advantage": mean_ci95(row["discovery_advantage"] for row in group),
            "critical_cost_ratio": {
                "n_finite": int(len(finite_rhos)),
                "median": float(np.median(finite_rhos)),
                "q10": float(np.quantile(finite_rhos, 0.10)),
                "q90": float(np.quantile(finite_rhos, 0.90)),
                "mean": float(finite_rhos.mean()),
            },
            "cost_ratio_frontier": ratio_summary,
        }

    return {
        "theoretical_bbp_time": float(runs[0]["theoretical_bbp_time"]),
        "by_time": by_time,
        "interpretation": (
            "Positive discovery advantage means the target-free wide-then-rank-one path beats the matched "
            "rank-one-from-start path at the same flow time. Positive break-even margin D-O means it also "
            "wins after giving the compact path extra time to match the declared wide/compact cost ratio. "
            "The critical cost ratio is the maximum rho that the observed discovery benefit can repay."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(20)))
    parser.add_argument("--dimension", type=int, default=64)
    parser.add_argument("--columns", type=int, default=128)
    parser.add_argument("--theta", type=float, default=1.0)
    parser.add_argument("--compact-horizon", type=float, default=0.2)
    parser.add_argument(
        "--checkpoints",
        type=float,
        nargs="+",
        default=[0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.60],
    )
    parser.add_argument(
        "--cost-ratios",
        type=float,
        nargs="+",
        default=[1.0, 2.0, 4.0, 8.0, 16.0],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/phase_discovery_frontier.json"),
    )
    args = parser.parse_args()

    if args.columns <= args.dimension:
        raise ValueError("columns must exceed dimension so the Wishart initialization is positive definite")
    if args.theta <= 0.0 or args.compact_horizon < 0.0:
        raise ValueError("theta must be positive and compact horizon non-negative")
    if any(t <= 0.0 for t in args.checkpoints):
        raise ValueError("all checkpoints must be strictly positive")
    if any(rho < 1.0 for rho in args.cost_ratios):
        raise ValueError("declared wide/compact cost ratios must be >= 1")

    runs = [
        run_seed(
            seed,
            dimension=args.dimension,
            columns=args.columns,
            theta=args.theta,
            compact_horizon=args.compact_horizon,
            checkpoints=args.checkpoints,
            cost_ratios=args.cost_ratios,
        )
        for seed in args.seeds
    ]
    payload = {
        "status": "precommitted_dynamic_bbp_discovery_frontier",
        "warning": (
            "The wide/compact cost ratios are a declared frontier, not a claim about hardware. The random "
            "matrix column count controls the statistical phase model and must not be silently equated with "
            "a neural accelerator cost ratio."
        ),
        "config": vars(args) | {"output": str(args.output)},
        "runs": runs,
        "summary": summarize(runs, args.cost_ratios),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
