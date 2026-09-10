"""Audit the theory-derived onset mechanism in the solvable quadratic model.

The experiment has no learned controller and no tuning.  It asks two narrow questions:

1. Does the exact finite-dimensional covariance flow show teacher/eigenvector resolution
   around the deterministic BBP crossing time from the theory?
2. What is the alignment-time penalty of a blind rank-one contraction at initialization
   versus retaining the teacher-associated outlier after spectral separation?

The second question uses the exact post-contraction angle law, not an optimizer proxy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from neural_net.quadratic_phase import (
    bbp_crossing_time,
    exact_covariance_flow,
    null_bulk_edges,
    rank_one_alignment_delay,
    top_teacher_overlap,
)


def run_one(d: int, gamma: float, theta: float, seed: int, epsilon: float):
    rng = np.random.default_rng(seed)
    m = int(round(d / gamma))
    g = rng.normal(size=(d, m))
    w0 = g / np.sqrt(m)
    a0 = w0 @ w0.T

    # Teacher orientation is independent of the Wishart initialization.
    u = rng.normal(size=d)
    u /= np.linalg.norm(u)
    a_star = theta * np.outer(u, u)

    tau = bbp_crossing_time(theta, gamma)
    time_multipliers = [0.0, 0.5, 0.9, 1.1, 1.5, 2.0, 3.0]
    trajectory = []
    post_overlap = None
    post_time = 2.0 * tau

    for c in time_multipliers:
        t = c * tau
        at = exact_covariance_flow(a0, a_star, t)
        top_eval, overlap = top_teacher_overlap(at, u)
        _, bulk_edge = null_bulk_edges(t, gamma)
        trajectory.append(
            {
                "time": t,
                "time_over_tau": c,
                "top_covariance_eigenvalue": top_eval,
                "bulk_upper_edge": bulk_edge,
                "top_minus_bulk_edge": top_eval - bulk_edge,
                "teacher_overlap": overlap,
            }
        )
        if abs(c - 2.0) < 1e-12:
            post_overlap = overlap

    # Blind rank-one contraction: choose the leading initialization eigenvector without
    # teacher information.  Independence makes its teacher overlap O_P(1/d).
    init_eval, init_overlap = top_teacher_overlap(a0, u)
    blind_delay = rank_one_alignment_delay(init_overlap, theta, epsilon)
    post_delay = rank_one_alignment_delay(post_overlap, theta, epsilon)

    return {
        "d": d,
        "m": m,
        "gamma_nominal": gamma,
        "gamma_finite": d / m,
        "theta": theta,
        "seed": seed,
        "epsilon": epsilon,
        "bbp_crossing_time": tau,
        "initial_top_eigenvalue": init_eval,
        "initial_teacher_overlap": init_overlap,
        "post_bbp_checkpoint": post_time,
        "post_bbp_teacher_overlap": post_overlap,
        "blind_alignment_delay": blind_delay,
        "post_bbp_alignment_delay": post_delay,
        "delay_reduction": blind_delay - post_delay,
        "trajectory": trajectory,
    }


def summarize(rows):
    by_d = {}
    for row in rows:
        by_d.setdefault(row["d"], []).append(row)
    summary = []
    for d, group in sorted(by_d.items()):
        arr = lambda key: np.asarray([g[key] for g in group], dtype=float)
        summary.append(
            {
                "d": d,
                "n_seeds": len(group),
                "mean_initial_overlap": float(arr("initial_teacher_overlap").mean()),
                "mean_d_times_initial_overlap": float(d * arr("initial_teacher_overlap").mean()),
                "mean_post_bbp_overlap": float(arr("post_bbp_teacher_overlap").mean()),
                "mean_blind_delay": float(arr("blind_alignment_delay").mean()),
                "mean_post_bbp_delay": float(arr("post_bbp_alignment_delay").mean()),
                "mean_delay_reduction": float(arr("delay_reduction").mean()),
                "mean_blind_delay_minus_logd_over_4theta": float(
                    (arr("blind_alignment_delay") - np.log(d) / (4.0 * group[0]["theta"])).mean()
                ),
            }
        )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dims", type=int, nargs="+", default=[40, 80, 160])
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=0.6)
    parser.add_argument("--theta", type=float, default=3.0)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--output", type=Path, default=Path("results/quadratic_onset_audit.json"))
    args = parser.parse_args()

    rows = [
        run_one(d, args.gamma, args.theta, seed, args.epsilon)
        for d in args.dims
        for seed in range(args.seeds)
    ]
    payload = {
        "config": vars(args) | {"output": str(args.output)},
        "summary": summarize(rows),
        "runs": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
