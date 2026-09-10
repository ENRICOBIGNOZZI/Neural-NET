from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import Ridge


def fit_ridge_readout(model, x: torch.Tensor, y: torch.Tensor, alpha: float = 1e-3):
    """Fit the model's scalar linear readout on frozen hidden features.

    The feature extractor is not changed.  The returned Ridge object is useful for
    diagnostics; the fitted coefficient and intercept are copied into ``model.out``.
    """
    was_training = model.training
    model.eval()
    with torch.no_grad():
        h = model.features(x).detach().cpu().numpy()
    target = y.detach().cpu().numpy().reshape(-1)
    reg = Ridge(alpha=float(alpha), fit_intercept=True)
    reg.fit(h, target)

    device = model.out.weight.device
    dtype = model.out.weight.dtype
    with torch.no_grad():
        coef = torch.as_tensor(np.asarray(reg.coef_).reshape(1, -1), device=device, dtype=dtype)
        intercept = torch.as_tensor([float(reg.intercept_)], device=device, dtype=dtype)
        model.out.weight.copy_(coef)
        model.out.bias.copy_(intercept)
    if was_training:
        model.train()
    return reg
