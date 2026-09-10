"""Coverage/tightness audit for structural-damage certificates.

The first version of this audit showed that the generic bounded-loss empirical-Bernstein
certificate has perfect coverage but is orders of magnitude too wide for the structural
effects of interest. This version separates three questions:

1. How wide is the fully distribution-free paired certificate?
2. How large is the unavoidable rare-event range barrier at the same probe size?
3. If an independent theory calculation supplied a valid second-moment bound, how much
   tighter could the truncated localized certificate become?

The second-moment quantities below are estimated from a very large independent Monte Carlo
sample only as an *oracle diagnostic*. They are not available to the controller and are not
reported as theorem-certified empirical evidence. Their role is to test whether deriving a
small geometry-based moment bound is worth pursuing.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from neural_net.models import ContractibleMLP
from neural_net.readout import fit_ridge_readout
from neural_net.spectral import target_greedy_neuron_order
from neural_net.structural_damage import (
    clipped_square_difference_second_moment_bound,
    clipped_squared_losses,
    distribution_free_zero_path_barrier,
    optimal_truncation_level,
    paired_damage_statistics,
    paired_empirical_bernstein_ucb,
    paired_truncated_second_moment_ucb,
)
from run_dense_vs_sra import features, step


def make_teacher(input_dim: int, seed: int):
    rng = np.random.default_rng(seed)
    w1 = rng.normal(size=input_dim)
    w2 = rng.normal(size=input_dim)
    w1 /= np.linalg.norm(w1)
    w2 -= w1 * (w1 @ w2)
    w2 /= np.linalg.norm(w2)

    def sample(n: int, sample_seed: int):
        rr = np.random.default_rng(sample_seed)
        x = rr.normal(size=(n, input_dim)).astype(np.float32)
        signal = 0.8 * np.tanh(x @ w1) + 0.45 * np.tanh(x @ w2)
        y = signal + 0.08 * rr.normal(size=n)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    return sample


def predict(model, x):
    model.eval()
    with torch.no_grad():
        return model(x).cpu().numpy()


def fit_branches(
    *,
    seed: int,
    input_dim: int,
    width: int,
    train_n: int,
    checkpoint: int,
    horizon: int,
    lr: float,
    ridge: float,
    keep_fractions: tuple[float, ...],
):
    sample = make_teacher(input_dim, seed + 9000)
    xtr, ytr = sample(train_n, seed + 100)

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = ContractibleMLP(input_dim, width, width)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(checkpoint):
        step(model, opt, xtr, ytr)

    snapshot = copy.deepcopy(model)
    htr = features(snapshot, xtr)
    order = target_greedy_neuron_order(htr, ytr.numpy())

    dense = copy.deepcopy(snapshot)
    fit_ridge_readout(dense, xtr, ytr, alpha=ridge)
    dense_opt = torch.optim.AdamW(dense.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(horizon):
        step(dense, dense_opt, xtr, ytr)

    reduced = []
    ranks = []
    for frac in keep_fractions:
        rank = max(1, min(width - 1, int(round(width * frac))))
        candidate = snapshot.contracted_copy(order[:rank])
        fit_ridge_readout(candidate, xtr, ytr, alpha=ridge)
        candidate_opt = torch.optim.AdamW(candidate.parameters(), lr=lr, weight_decay=1e-4)
        for _ in range(horizon):
            step(candidate, candidate_opt, xtr, ytr)
        reduced.append(candidate)
        ranks.append(rank)
    return sample, dense, reduced, ranks


def evaluate(sample, dense, reduced, *, n: int, seed: int, loss_bound: float):
    x, y = sample(n, seed)
    target = y.numpy()
    dense_pred = predict(dense, x)
    reduced_pred = [predict(model, x) for model in reduced]
    dense_loss = clipped_squared_losses(dense_pred, target, loss_bound=loss_bound)
    reduced_loss = [
        clipped_squared_losses(pred, target, loss_bound=loss_bound)
        for pred in reduced_pred
    ]
    return dense_loss, reduced_loss, dense_pred, reduced_pred


def run_seed(args, seed: int):
    sample, dense, reduced, ranks = fit_branches(
        seed=seed,
        input_dim=args.input_dim,
        width=args.width,
        train_n=args.train_n,
        checkpoint=args.checkpoint,
        horizon=args.horizon,
        lr=args.lr,
        ridge=args.ridge,
        keep_fractions=tuple(args.keep_fractions),
    )
    m = len(reduced)

    dense_pop, reduced_pop, dense_pred_pop, reduced_pred_pop = evaluate(
        sample,
        dense,
        reduced,
        n=args.population_n,
        seed=seed + 500_000,
        loss_bound=args.loss_bound,
    )
    population_z = [r - dense_pop for r in reduced_pop]
    true_damage = np.asarray([float(np.mean(z)) for z in population_z], dtype=float)
    oracle_pair_m2 = np.asarray([float(np.mean(z**2)) for z in population_z], dtype=float)
    oracle_prediction_l2 = np.asarray(
        [float(np.sqrt(np.mean((rp - dense_pred_pop) ** 2))) for rp in reduced_pred_pop],
        dtype=float,
    )
    oracle_geometry_m2 = np.asarray(
        [
            clipped_square_difference_second_moment_bound(
                loss_bound=args.loss_bound,
                prediction_l2_bound=d,
            )
            for d in oracle_prediction_l2
        ],
        dtype=float,
    )

    # The large Monte Carlo sample is only an approximation to population quantities. A
    # modest inflation avoids pretending that the oracle diagnostic is exact. These bounds
    # are still *not* controller-valid; a real algorithm must derive them without this sample.
    oracle_pair_m2_for_audit = np.minimum(args.loss_bound**2, args.oracle_inflation * oracle_pair_m2)
    oracle_geometry_m2_for_audit = np.minimum(
        args.loss_bound**2,
        args.oracle_inflation * oracle_geometry_m2,
    )

    by_n = []
    for n in args.probe_sizes:
        simultaneous_cover_generic = []
        simultaneous_cover_pair_oracle = []
        simultaneous_cover_geometry_oracle = []
        candidate_cover_generic = []
        candidate_cover_pair_oracle = []
        candidate_cover_geometry_oracle = []
        generic_slack = []
        pair_oracle_slack = []
        geometry_oracle_slack = []
        empirical_damage = []

        pair_c = [
            optimal_truncation_level(
                loss_bound=args.loss_bound,
                second_moment_bound=m2,
                n=n,
                delta=args.delta,
                num_candidates=m,
            )
            for m2 in oracle_pair_m2_for_audit
        ]
        geometry_c = [
            optimal_truncation_level(
                loss_bound=args.loss_bound,
                second_moment_bound=m2,
                n=n,
                delta=args.delta,
                num_candidates=m,
            )
            for m2 in oracle_geometry_m2_for_audit
        ]

        for rep in range(args.repetitions):
            dense_loss, reduced_loss, _, _ = evaluate(
                sample,
                dense,
                reduced,
                n=n,
                seed=seed * 1_000_000 + n * 10_000 + rep,
                loss_bound=args.loss_bound,
            )
            means = []
            generic_ucbs = []
            pair_oracle_ucbs = []
            geometry_oracle_ucbs = []
            for j, rloss in enumerate(reduced_loss):
                mean, _, _ = paired_damage_statistics(rloss, dense_loss)
                generic = paired_empirical_bernstein_ucb(
                    rloss,
                    dense_loss,
                    loss_bound=args.loss_bound,
                    delta=args.delta,
                    num_candidates=m,
                )
                pair_oracle = paired_truncated_second_moment_ucb(
                    rloss,
                    dense_loss,
                    loss_bound=args.loss_bound,
                    second_moment_bound=float(oracle_pair_m2_for_audit[j]),
                    delta=args.delta,
                    num_candidates=m,
                )
                geometry_oracle = paired_truncated_second_moment_ucb(
                    rloss,
                    dense_loss,
                    loss_bound=args.loss_bound,
                    second_moment_bound=float(oracle_geometry_m2_for_audit[j]),
                    delta=args.delta,
                    num_candidates=m,
                )
                means.append(mean)
                generic_ucbs.append(generic)
                pair_oracle_ucbs.append(pair_oracle)
                geometry_oracle_ucbs.append(geometry_oracle)

            means = np.asarray(means)
            generic_ucbs = np.asarray(generic_ucbs)
            pair_oracle_ucbs = np.asarray(pair_oracle_ucbs)
            geometry_oracle_ucbs = np.asarray(geometry_oracle_ucbs)

            cover_generic = true_damage <= generic_ucbs
            cover_pair_oracle = true_damage <= pair_oracle_ucbs
            cover_geometry_oracle = true_damage <= geometry_oracle_ucbs
            simultaneous_cover_generic.append(bool(np.all(cover_generic)))
            simultaneous_cover_pair_oracle.append(bool(np.all(cover_pair_oracle)))
            simultaneous_cover_geometry_oracle.append(bool(np.all(cover_geometry_oracle)))
            candidate_cover_generic.append(cover_generic.astype(float))
            candidate_cover_pair_oracle.append(cover_pair_oracle.astype(float))
            candidate_cover_geometry_oracle.append(cover_geometry_oracle.astype(float))
            generic_slack.append(generic_ucbs - means)
            pair_oracle_slack.append(pair_oracle_ucbs - means)
            geometry_oracle_slack.append(geometry_oracle_ucbs - means)
            empirical_damage.append(means)

        by_n.append(
            {
                "probe_n": int(n),
                "generic_simultaneous_coverage": float(np.mean(simultaneous_cover_generic)),
                "oracle_pair_moment_simultaneous_coverage": float(np.mean(simultaneous_cover_pair_oracle)),
                "oracle_geometry_moment_simultaneous_coverage": float(np.mean(simultaneous_cover_geometry_oracle)),
                "generic_candidate_coverage": np.mean(candidate_cover_generic, axis=0).tolist(),
                "oracle_pair_moment_candidate_coverage": np.mean(candidate_cover_pair_oracle, axis=0).tolist(),
                "oracle_geometry_moment_candidate_coverage": np.mean(candidate_cover_geometry_oracle, axis=0).tolist(),
                "mean_generic_slack": np.mean(generic_slack, axis=0).tolist(),
                "mean_oracle_pair_moment_slack": np.mean(pair_oracle_slack, axis=0).tolist(),
                "mean_oracle_geometry_moment_slack": np.mean(geometry_oracle_slack, axis=0).tolist(),
                "median_generic_slack": np.median(generic_slack, axis=0).tolist(),
                "median_oracle_pair_moment_slack": np.median(pair_oracle_slack, axis=0).tolist(),
                "median_oracle_geometry_moment_slack": np.median(geometry_oracle_slack, axis=0).tolist(),
                "mean_empirical_damage": np.mean(empirical_damage, axis=0).tolist(),
                "generic_zero_path_barrier_bonferroni": distribution_free_zero_path_barrier(
                    range_bound=args.loss_bound,
                    n=n,
                    delta=args.delta / m,
                ),
                "oracle_pair_truncation_levels": pair_c,
                "oracle_geometry_truncation_levels": geometry_c,
            }
        )

    return {
        "seed": seed,
        "candidate_ranks": ranks,
        "true_clipped_population_damage": true_damage.tolist(),
        "oracle_pair_second_moment": oracle_pair_m2.tolist(),
        "oracle_prediction_l2_gap": oracle_prediction_l2.tolist(),
        "oracle_geometry_second_moment_bound": oracle_geometry_m2.tolist(),
        "probe_results": by_n,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--input-dim", type=int, default=20)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--train-n", type=int, default=1200)
    parser.add_argument("--checkpoint", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=40)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--keep-fractions", type=float, nargs="+", default=[0.75, 0.875])
    parser.add_argument("--population-n", type=int, default=50000)
    parser.add_argument("--probe-sizes", type=int, nargs="+", default=[64, 128, 256, 512, 1024])
    parser.add_argument("--repetitions", type=int, default=200)
    parser.add_argument("--loss-bound", type=float, default=4.0)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--oracle-inflation", type=float, default=1.10)
    parser.add_argument("--output", type=Path, default=Path("results/damage_ucb_coverage.json"))
    args = parser.parse_args()

    rows = [run_seed(args, seed) for seed in args.seeds]

    aggregate = []
    for i, n in enumerate(args.probe_sizes):
        aggregate.append(
            {
                "probe_n": int(n),
                "mean_generic_simultaneous_coverage": float(
                    np.mean([r["probe_results"][i]["generic_simultaneous_coverage"] for r in rows])
                ),
                "mean_oracle_pair_moment_simultaneous_coverage": float(
                    np.mean([r["probe_results"][i]["oracle_pair_moment_simultaneous_coverage"] for r in rows])
                ),
                "mean_oracle_geometry_moment_simultaneous_coverage": float(
                    np.mean([r["probe_results"][i]["oracle_geometry_moment_simultaneous_coverage"] for r in rows])
                ),
                "mean_generic_slack": np.mean(
                    [r["probe_results"][i]["mean_generic_slack"] for r in rows], axis=0
                ).tolist(),
                "mean_oracle_pair_moment_slack": np.mean(
                    [r["probe_results"][i]["mean_oracle_pair_moment_slack"] for r in rows], axis=0
                ).tolist(),
                "mean_oracle_geometry_moment_slack": np.mean(
                    [r["probe_results"][i]["mean_oracle_geometry_moment_slack"] for r in rows], axis=0
                ).tolist(),
                "mean_zero_path_barrier": float(
                    np.mean([r["probe_results"][i]["generic_zero_path_barrier_bonferroni"] for r in rows])
                ),
                "mean_abs_true_damage": float(
                    np.mean(np.abs([d for r in rows for d in r["true_clipped_population_damage"]]))
                ),
                "mean_oracle_pair_second_moment": float(
                    np.mean([d for r in rows for d in r["oracle_pair_second_moment"]])
                ),
                "mean_oracle_geometry_second_moment_bound": float(
                    np.mean([d for r in rows for d in r["oracle_geometry_second_moment_bound"]])
                ),
            }
        )

    payload = {
        "config": vars(args) | {"output": str(args.output)},
        "aggregate": aggregate,
        "runs": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
