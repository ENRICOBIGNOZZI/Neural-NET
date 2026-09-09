"""Numerical audit of the exact fixed-geometry reduction theorem."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from neural_net.spectral import fixed_geometry_mode_value


def main():
    rng = np.random.default_rng(7)
    p = 40
    mu = np.geomspace(5.0, 1e-4, p)
    a = rng.normal(size=p) / np.sqrt(np.arange(1, p + 1))
    horizon = 10.0
    exact_value = fixed_geometry_mode_value(mu, a, horizon)

    keep = exact_value >= np.quantile(exact_value, 0.5)
    full_terminal = 0.5 * np.sum(a**2 * np.exp(-2 * mu * horizon))
    reduced_terminal = 0.5 * np.sum(
        np.where(keep, a**2 * np.exp(-2 * mu * horizon), a**2)
    )
    identity_error = abs(
        (reduced_terminal - full_terminal) - float(exact_value[~keep].sum())
    )

    out = {
        "horizon": horizon,
        "modes": p,
        "kept_modes": int(keep.sum()),
        "terminal_risk_full": float(full_terminal),
        "terminal_risk_reduced": float(reduced_terminal),
        "identity_error": float(identity_error),
    }
    Path("results").mkdir(exist_ok=True)
    Path("results/fixed_geometry_audit.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
