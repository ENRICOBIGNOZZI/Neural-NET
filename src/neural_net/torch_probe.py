from __future__ import annotations

import numpy as np
import torch

from .spectral import tangent_spectrum_from_jacobian


def empirical_output_jacobian(model: torch.nn.Module, x: torch.Tensor) -> np.ndarray:
    """Dense per-example output Jacobian for small diagnostic experiments only."""
    params = [p for p in model.parameters() if p.requires_grad]
    rows = []
    was_training = model.training
    model.eval()
    for i in range(x.shape[0]):
        model.zero_grad(set_to_none=True)
        out = model(x[i : i + 1]).reshape(())
        grads = torch.autograd.grad(out, params, retain_graph=False, create_graph=False)
        rows.append(torch.cat([g.reshape(-1) for g in grads]).detach().cpu().numpy())
    if was_training:
        model.train()
    return np.stack(rows)


def empirical_tangent_spectrum(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor):
    jac = empirical_output_jacobian(model, x)
    with torch.no_grad():
        residual = (y - model(x)).detach().cpu().numpy()
    return tangent_spectrum_from_jacobian(jac, residual)
