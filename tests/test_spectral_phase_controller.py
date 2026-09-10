from __future__ import annotations

import numpy as np

from neural_net.spectral_phase_controller import (
    DualSpectrumPhaseController,
    SpectralPhaseConfig,
    target_free_spectral_diagnostics,
)


def test_phase_certificate_accepts_clean_two_block_spectrum():
    top = np.linspace(2.4, 1.4, 12)
    bottom = np.linspace(0.7, 0.3, 12)
    decision = DualSpectrumPhaseController().evaluate_eigenvalues(
        np.concatenate([top, bottom])
    )
    assert decision.should_contract
    assert decision.proposed_rank == 12
    assert decision.largest_gap_rank == 12
    assert decision.logsplit_rank == 12
    assert decision.boundary_ratio >= 1.5
    assert decision.retained_mass_fraction <= 0.90


def test_phase_certificate_rejects_vanishing_tail_split():
    top = np.linspace(1.2, 0.8, 20)
    tail = np.full(4, 1e-4)
    decision = DualSpectrumPhaseController().evaluate_eigenvalues(
        np.concatenate([top, tail])
    )
    assert not decision.should_contract
    assert decision.proposed_rank == 24
    assert "candidate split only removes a vanishing spectral tail" in decision.reasons


def test_phase_certificate_rejects_unseparated_boundary():
    top = np.linspace(1.4, 1.05, 12)
    bottom = np.linspace(0.95, 0.65, 12)
    decision = DualSpectrumPhaseController().evaluate_eigenvalues(
        np.concatenate([top, bottom])
    )
    assert not decision.should_contract
    assert decision.proposed_rank == 24
    assert "spectral boundary is not separated enough" in decision.reasons


def test_diagnostics_sort_and_clip_eigenvalues():
    values = np.array([0.2, -1e-9, 2.0, 0.5, 1.0])
    diagnostics = target_free_spectral_diagnostics(values)
    ordered = diagnostics["eigenvalues_desc"]
    assert np.all(ordered[:-1] >= ordered[1:])
    assert ordered[-1] == 0.0


def test_thresholds_are_configurable_but_frozen_by_default():
    default = SpectralPhaseConfig()
    assert default.min_boundary_ratio == 1.5
    assert default.max_retained_mass_fraction == 0.90
