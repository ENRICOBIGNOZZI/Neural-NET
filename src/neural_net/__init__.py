"""Safe spectral reduction research code."""

from .rank_factor import (
    RankContractibleMLP,
    RankFactorLinear,
    deleted_scale_norm,
    physical_rank_parameter_count,
    rank_factor_group_gradient_energy,
)
from .readout import fit_ridge_readout
from .safe_certificate import (
    checkpoint_prediction_gap_certificate,
    structural_group_prediction_gap_certificate,
)
from .spectral import (
    accessibility_from_features,
    effective_rank,
    fixed_geometry_mode_value,
    fixed_geometry_subset_cost,
    grassmann_distance,
    sequence_drop_delta,
)

__all__ = [
    "fit_ridge_readout",
    "accessibility_from_features",
    "effective_rank",
    "fixed_geometry_mode_value",
    "fixed_geometry_subset_cost",
    "grassmann_distance",
    "sequence_drop_delta",
    "RankFactorLinear",
    "RankContractibleMLP",
    "deleted_scale_norm",
    "physical_rank_parameter_count",
    "rank_factor_group_gradient_energy",
    "checkpoint_prediction_gap_certificate",
    "structural_group_prediction_gap_certificate",
]
