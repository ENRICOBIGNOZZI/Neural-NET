from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from run_quadratic_factor_bridge import mean_simultaneous_ci


EXPECTED_ARCHITECTURES = {24, 30, 36, 42, 48}
SIMULTANEOUS_TESTS = 15  # 5 architectures x {S, D, M}
MIN_TRIGGERED_FOR_CI = 20


def ci_or_none(values: list[float]) -> dict | None:
    if len(values) < MIN_TRIGGERED_FOR_CI:
        return None
    return mean_simultaneous_ci(
        values,
        alpha=0.05,
        simultaneous_tests=SIMULTANEOUS_TESTS,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payloads = [json.loads(path.read_text()) for path in args.inputs]
    observed = {int(p["config"]["wide_factors"]) for p in payloads}
    if observed != EXPECTED_ARCHITECTURES or len(payloads) != 5:
        raise ValueError(
            f"expected exactly architectures {sorted(EXPECTED_ARCHITECTURES)}, "
            f"got {sorted(observed)}"
        )

    architectures = {}
    all_runs = []
    passing_architectures = []
    for payload in sorted(payloads, key=lambda p: int(p["config"]["wide_factors"])):
        wide_factors = int(payload["config"]["wide_factors"])
        runs = payload["runs"]
        if len(runs) != 50 or {int(r["seed"]) for r in runs} != set(range(300, 350)):
            raise ValueError(
                f"architecture {wide_factors} does not contain frozen seeds 300--349"
            )
        all_runs.extend(
            [{"wide_factors": wide_factors, **run} for run in runs]
        )
        triggered = [run for run in runs if run["triggered"]]

        s_values = [float(run["future_structural_damage_S"]) for run in triggered]
        d_values = [float(run["discovery_advantage_D"]) for run in triggered]
        m_values = [float(run["equal_compute_margin_M"]) for run in triggered]
        o_values = [float(run["compute_opportunity_gain_O"]) for run in triggered]
        s_ci = ci_or_none(s_values)
        d_ci = ci_or_none(d_values)
        m_ci = ci_or_none(m_values)

        safe = bool(s_ci is not None and s_ci["ci_hi"] <= 0.0)
        discovery_positive = bool(d_ci is not None and d_ci["ci_lo"] > 0.0)
        equal_compute_positive = bool(m_ci is not None and m_ci["ci_lo"] > 0.0)
        architecture_pass = bool(
            safe and discovery_positive and equal_compute_positive
        )
        if architecture_pass:
            passing_architectures.append(wide_factors)

        delays = [
            int(run["trigger_delay_from_selected_oracle_safe"])
            for run in triggered
            if run["trigger_delay_from_selected_oracle_safe"] is not None
        ]
        selected_ranks = Counter(int(run["selected_rank"]) for run in triggered)
        trigger_checkpoints = Counter(
            int(run["trigger_checkpoint"]) for run in triggered
        )

        architectures[str(wide_factors)] = {
            "wide_factors": wide_factors,
            "runs": len(runs),
            "triggered": len(triggered),
            "trigger_rate": float(len(triggered) / len(runs)),
            "selected_rank_counts": {
                str(k): int(v) for k, v in sorted(selected_ranks.items())
            },
            "trigger_checkpoint_counts": {
                str(k): int(v) for k, v in sorted(trigger_checkpoints.items())
            },
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
            "median_trigger_delay": (
                float(np.median(delays)) if delays else None
            ),
            "max_trigger_delay": max(delays) if delays else None,
            "mean_compute_opportunity_gain_O": (
                float(np.mean(o_values)) if o_values else None
            ),
            "future_structural_damage_S": s_ci,
            "discovery_advantage_D": d_ci,
            "equal_compute_margin_M": m_ci,
            "simultaneously_safe": safe,
            "simultaneously_positive_discovery": discovery_positive,
            "simultaneously_positive_equal_compute": equal_compute_positive,
            "architecture_pass": architecture_pass,
        }

    triggered_all = [run for run in all_runs if run["triggered"]]
    under_total = int(
        sum(run["under_ranked_teacher"] for run in triggered_all)
    )
    false_early_total = int(
        sum(
            run["trigger_before_selected_oracle_safe"]
            for run in triggered_all
        )
    )
    over_total = int(
        sum(run["over_ranked_teacher"] for run in triggered_all)
    )
    exact_total = int(
        sum(run["rank_equals_teacher"] for run in triggered_all)
    )
    missed_total = int(
        sum(
            run.get("missed_teacher_safe_opportunity", False)
            for run in all_runs
        )
    )

    result = {
        "status": "precommitted_nonoracle_controller_confirmation_aggregate",
        "family": {
            "development_seeds": "0--49",
            "confirmatory_seeds": "300--349",
            "architectures": [24, 30, 36, 42, 48],
            "checkpoints": [20, 40, 60, 80, 100, 120],
            "controller": {
                "rank_agreement": "largest-gap rank equals log-spectrum two-cluster rank",
                "min_boundary_ratio": 1.5,
                "max_retained_mass_fraction": 0.90,
                "stopping": "first checkpoint satisfying all gates",
            },
            "inference": (
                "Bonferroni two-sided 95% paired Student-t intervals over the "
                "15 architecture-by-estimand tests for S, D, and M; an "
                f"architecture needs at least {MIN_TRIGGERED_FOR_CI} triggered "
                "seeds for an inferential claim."
            ),
        },
        "headline": {
            "total_runs": len(all_runs),
            "total_triggers": len(triggered_all),
            "overall_trigger_rate": float(
                len(triggered_all) / len(all_runs)
            ),
            "exact_teacher_rank_triggers": exact_total,
            "under_rank_events": under_total,
            "over_rank_events": over_total,
            "trigger_before_selected_oracle_safe_events": false_early_total,
            "missed_teacher_safe_opportunities": missed_total,
            "passing_architectures": passing_architectures,
            "target_free_rank_safety_pass": bool(under_total == 0),
            "onset_safety_pass": bool(false_early_total == 0),
            "nonoracle_physical_equal_compute_regime": bool(
                under_total == 0
                and false_early_total == 0
                and passing_architectures
            ),
        },
        "architectures": architectures,
        "interpretation": (
            "The controller never sees teacher rank, teacher subspace, future "
            "loss, oracle safe time, or equal-compute outcomes. A family-level "
            "pass requires no destructive under-ranking, no trigger before the "
            "evaluation-only safe time for the selected structural action, and "
            "at least one architecture whose controller-selected contraction "
            "is simultaneously safe, discovery-positive, and equal-compute "
            "positive. No-trigger runs are abstentions rather than forced "
            "contractions."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["headline"], indent=2))


if __name__ == "__main__":
    main()
