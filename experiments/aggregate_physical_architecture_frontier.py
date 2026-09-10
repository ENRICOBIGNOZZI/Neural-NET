from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payloads = [json.loads(path.read_text()) for path in args.inputs]
    expected_ranks = {24, 30, 36, 42, 48}
    observed_ranks = {int(p["config"]["wide_factors"]) for p in payloads}
    if observed_ranks != expected_ranks or len(payloads) != len(expected_ranks):
        raise ValueError(
            f"expected exactly ranks {sorted(expected_ranks)}, got {sorted(observed_ranks)}"
        )

    architectures = {}
    positive_m_cells = []
    positive_d_cells = []
    safe_cells = []
    all_cells = []
    for payload in sorted(payloads, key=lambda p: int(p["config"]["wide_factors"])):
        rank = int(payload["config"]["wide_factors"])
        ratio = float(payload["wide_to_compact_mac_ratio"])
        summary = payload["summary"]
        architectures[str(rank)] = {
            "wide_factors": rank,
            "mac_ratio": ratio,
            "earliest_safe_checkpoint": summary["earliest_safe_checkpoint"],
            "earliest_positive_discovery_checkpoint": summary[
                "earliest_positive_discovery_checkpoint"
            ],
            "earliest_positive_equal_compute_checkpoint": summary[
                "earliest_positive_equal_compute_checkpoint"
            ],
            "best_mean_equal_compute_checkpoint": summary[
                "best_mean_equal_compute_checkpoint"
            ],
            "by_checkpoint": summary["by_checkpoint"],
        }
        for checkpoint, cell in summary["by_checkpoint"].items():
            checkpoint_int = int(checkpoint)
            row = {
                "wide_factors": rank,
                "mac_ratio": ratio,
                "checkpoint": checkpoint_int,
                "S": cell["future_structural_damage"],
                "D": cell["discovery_advantage_D"],
                "O": cell["compute_opportunity_gain_O"],
                "M": cell["equal_compute_margin_M"],
                "overlap": cell["teacher_subspace_overlap"],
                "eigengap": cell["eigengap"],
                "bulk_trace_fraction": cell["bulk_trace_fraction"],
                "wide_win_fraction": cell["wide_win_fraction"],
            }
            all_cells.append(row)
            if cell["equal_compute_margin_M"]["ci_lo"] > 0.0:
                positive_m_cells.append(row)
            if cell["discovery_advantage_D"]["ci_lo"] > 0.0:
                positive_d_cells.append(row)
            if cell["future_structural_damage"]["ci_hi"] <= 0.0:
                safe_cells.append(row)

    best_mean = max(all_cells, key=lambda row: row["M"]["mean"])
    best_lower = max(all_cells, key=lambda row: row["M"]["ci_lo"])
    result = {
        "status": "precommitted_physical_architecture_frontier_aggregate",
        "family": {
            "seeds": "200--249",
            "architectures": [24, 30, 36, 42, 48],
            "checkpoints": [20, 40, 60, 80, 100, 120],
            "cells": 30,
            "simultaneous_alpha": 0.05,
            "correction": "Bonferroni two-sided Student-t across all 30 frozen cells",
        },
        "headline": {
            "mechanism_positive": bool(positive_d_cells),
            "safe_contraction_observed": bool(safe_cells),
            "physical_equal_compute_regime": bool(positive_m_cells),
            "number_positive_equal_compute_cells": len(positive_m_cells),
            "best_mean_equal_compute_cell": best_mean,
            "best_simultaneous_lower_bound_cell": best_lower,
        },
        "positive_equal_compute_cells": positive_m_cells,
        "architectures": architectures,
        "all_cells": all_cells,
        "interpretation": (
            "A positive physical equal-compute regime requires a 30-cell simultaneous lower confidence bound for M above zero. "
            "Changing wide rank also changes the Wishart aspect ratio, so cross-rank differences are physical architecture differences rather than a pure cost-price experiment."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["headline"], indent=2))


if __name__ == "__main__":
    main()
