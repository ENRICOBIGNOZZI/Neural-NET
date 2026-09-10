from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class SpectralPhaseConfig:
    """Frozen target-free phase certificate selected on development seeds 0--49."""

    min_boundary_ratio: float = 1.5
    max_retained_mass_fraction: float = 0.90


@dataclass(frozen=True)
class SpectralPhaseDecision:
    should_contract: bool
    proposed_rank: int
    largest_gap_rank: int
    logsplit_rank: int
    boundary_ratio: float
    retained_mass_fraction: float
    gap_dominance: float
    logsplit_separation: float
    reasons: tuple[str, ...] = field(default_factory=tuple)


def target_free_spectral_diagnostics(eigenvalues_desc) -> dict:
    """Compute the two development rank detectors from eigenvalues only.

    Parameters
    ----------
    eigenvalues_desc:
        Current covariance eigenvalues. Values are sorted internally in descending
        order and clipped at zero before diagnostics are formed.
    """

    vals = np.asarray(eigenvalues_desc, dtype=float).reshape(-1)
    if vals.size < 3:
        raise ValueError("spectral phase detector requires at least three eigenvalues")
    if not np.all(np.isfinite(vals)):
        raise ValueError("eigenvalues must be finite")

    vals = np.maximum(np.sort(vals)[::-1], 0.0)
    d = int(vals.size)
    scale = max(float(vals[0]), 1.0)
    eps = 1e-12 * scale

    gaps = vals[:-1] - vals[1:]
    order = np.argsort(gaps)[::-1]
    best = int(order[0])
    second_gap = float(gaps[order[1]]) if len(order) > 1 else 0.0
    largest_gap = float(gaps[best])
    largest_gap_rank = best + 1
    gap_dominance = largest_gap / max(second_gap, eps)

    logvals = np.log(vals + eps)
    split_scores: list[float] = []
    for k in range(1, d):
        top = logvals[:k]
        bottom = logvals[k:]
        sse = float(
            np.sum((top - top.mean()) ** 2)
            + np.sum((bottom - bottom.mean()) ** 2)
        )
        split_scores.append(sse)
    logsplit_rank = int(np.argmin(split_scores)) + 1
    k = logsplit_rank
    top = logvals[:k]
    bottom = logvals[k:]
    pooled = np.sqrt(float(np.var(top) + np.var(bottom)) + 1e-12)
    logsplit_separation = float((top.mean() - bottom.mean()) / pooled)

    boundary_ratio = float(vals[k - 1] / max(vals[k], eps))
    total = float(vals.sum())
    retained_mass_fraction = (
        float(vals[:k].sum() / total) if total > 0.0 else 0.0
    )

    return {
        "eigenvalues_desc": vals,
        "largest_gap_rank": int(largest_gap_rank),
        "logsplit_rank": int(logsplit_rank),
        "largest_gap": largest_gap,
        "second_largest_gap": second_gap,
        "gap_dominance": float(gap_dominance),
        "boundary_ratio": boundary_ratio,
        "retained_mass_fraction": retained_mass_fraction,
        "logsplit_separation": logsplit_separation,
    }


class DualSpectrumPhaseController:
    """Target-free structural-onset controller.

    The controller contracts only when two independent spectral rank detectors
    identify the same boundary, that boundary has at least a 1.5 eigenvalue
    ratio, and the proposed retained cluster carries at most 90% of current
    spectral mass. The mass gate prevents a numerically tiny tail from being
    mistaken for a meaningful phase boundary.
    """

    def __init__(self, config: SpectralPhaseConfig | None = None):
        self.config = config or SpectralPhaseConfig()

    def evaluate_eigenvalues(self, eigenvalues_desc) -> SpectralPhaseDecision:
        cfg = self.config
        d = target_free_spectral_diagnostics(eigenvalues_desc)
        k_gap = int(d["largest_gap_rank"])
        k_log = int(d["logsplit_rank"])
        reasons: list[str] = []

        if k_gap != k_log:
            reasons.append("rank detectors disagree")
        if d["boundary_ratio"] < cfg.min_boundary_ratio:
            reasons.append("spectral boundary is not separated enough")
        if d["retained_mass_fraction"] > cfg.max_retained_mass_fraction:
            reasons.append("candidate split only removes a vanishing spectral tail")

        should = not reasons
        proposed_rank = k_log if should else len(d["eigenvalues_desc"])
        return SpectralPhaseDecision(
            should_contract=bool(should),
            proposed_rank=int(proposed_rank),
            largest_gap_rank=k_gap,
            logsplit_rank=k_log,
            boundary_ratio=float(d["boundary_ratio"]),
            retained_mass_fraction=float(d["retained_mass_fraction"]),
            gap_dominance=float(d["gap_dominance"]),
            logsplit_separation=float(d["logsplit_separation"]),
            reasons=tuple(reasons),
        )
