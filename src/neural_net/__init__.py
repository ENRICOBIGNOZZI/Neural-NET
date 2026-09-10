"""Safe spectral reduction research code."""

from .bulk_readiness import BulkReadinessCertificate, bulk_is_geometrically_ready, bulk_readiness_certificate
from .bulk_spectrum import BulkSpectralState, bulk_state_from_tangent_spectrum
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
    "BulkReadinessCertificate",
    "bulk_readiness_certificate",
    "bulk_is_geometrically_ready",
    "BulkSpectralState",
    "bulk_state_from_tangent_spectrum",
]
