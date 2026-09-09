"""Coverage/tightness audit for the paired structural-damage certificate.

This experiment is intentionally synthetic so that a very large independent Monte Carlo
sample can approximate the population clipped-loss damage of each declared structural
action.  Candidate masks are built from training data only.  The independent evaluation
sample is never used to construct the candidates or train the branches.

For each probe size n we repeatedly draw fresh probe samples and check whether the
simultaneous empirical-Bernstein UCB covers the large-Monte-Carlo population damage for
all candidates.  This tests the finite-sample *certificate*, not controller performance.
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
    clipped_squared_losses,
    paired_damage_statistics,
    paired_empirical_bernstein_ucb,
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


def evaluate_losses(sample, dense, reduced, *, n: int, seed: int, loss_bound: float):
    x, y = sample(n, seed)
    target = y.numpy()
    dense_loss = clipped_squared_losses(predict(dense, x), target, loss_bound=loss_bound)
    reduced_loss = [
        clipped_squared_losses(predict(model, x), target, loss_bound=loss_bound)
        for model in reduced
    ]
    return dense_loss, reduced_loss


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

    dense_pop, reduced_pop = evaluate_losses(
        sample,
        dense,
        reduced,
        n=args.population_n,
        seed=seed + 500_000,
        loss_bound=args.loss_bound,
    )
    true_damage = np.asarray(
        [float(np.mean(r - dense_pop)) for r in reduced_pop],
        dtype=float,
    )

    by_n = []
    for n in args.probe_sizes:
        simultaneous_cover = []
        candidate_cover = []
        widths = []
        empirical_damage = []
        for rep in range(args.repetitions):
            dense_loss, reduced_loss = evaluate_losses(
                sample,
                dense,
                reduced,
                n=n,
                seed=seed * 1_000_000 + n * 10_000 + rep,
                loss_bound=args.loss_bound,
            )
            ucbs = []
            means = []
            for rloss in reduced_loss:
                mean, _, _ = paired_damage_statistics(rloss, dense_loss)
                ucb = paired_empirical_bernstein_ucb(
                    rloss,
                    dense_loss,
                    loss_bound=args.loss_bound,
                    delta=args.delta,
                    num_candidates=m,
                )
                means.append(mean)
                ucbs.append(ucb)
            ucbs = np.asarray(ucbs)
            means = np.asarray(means)
            cover = true_damage <= ucbs
            simultaneous_cover.append(bool(np.all(cover)))
            candidate_cover.append(cover.astype(float))
            widths.append(ucbs - means)
            empirical_damage.append(means)

        by_n.append(
            {
                "probe_n": int(n),
                "simultaneous_coverage": float(np.mean(simultaneous_cover)),
                "candidate_coverage": np.mean(candidate_cover, axis=0).tolist(),
                "mean_ucb_radius": np.mean(widths, axis=0).tolist(),
                "median_ucb_radius": np.median(widths, axis=0).tolist(),
                "mean_empirical_damage": np.mean(empirical_damage, axis=0).tolist(),
            }
        )

    return {
        "seed": seed,
        "candidate_ranks": ranks,
        "true_clipped_population_damage": true_damage.tolist(),
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
    parser.add_argument("--output", type=Path, default=Path("results/damage_ucb_coverage.json"))
    args = parser.parse_args()

    rows = [run_seed(args, seed) for seed in args.seeds]

    aggregate = []
    for i, n in enumerate(args.probe_sizes):
        aggregate.append(
            {
                "probe_n": int(n),
                "mean_simultaneous_coverage": float(np.mean([r["probe_results"][i]["simultaneous_coverage"] for r in rows])),
                "mean_ucb_radius": np.mean([r["probe_results"][i]["mean_ucb_radius"] for r in rows], axis=0).tolist(),
                "mean_abs_true_damage": float(np.mean(np.abs([d for r in rows for d in r["true_clipped_population_damage"]]))),
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
