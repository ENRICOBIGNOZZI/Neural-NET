from __future__ import annotations

import copy
import math

import numpy as np
import torch
from torch import nn


class BilinearQuadraticFactor(nn.Module):
    """Factor network ``f_W(x,z) = x' W W' z``.

    For independent isotropic inputs x and z and target ``x' A* z``, population
    half-MSE is exactly ``0.5 ||W W' - A*||_F^2``. Consequently population
    gradient flow in W induces the exact covariance Riccati dynamics used by the
    quadratic phase theorem.
    """

    def __init__(
        self,
        dimension: int,
        factors: int,
        *,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if dimension < 1 or factors < 1:
            raise ValueError("dimension and factors must be positive")
        self.dimension = int(dimension)
        self.factors = int(factors)
        w = torch.empty(self.dimension, self.factors, dtype=dtype, device=device)
        nn.init.normal_(w, mean=0.0, std=1.0 / math.sqrt(self.factors))
        self.weight = nn.Parameter(w)

    @classmethod
    def from_weight(cls, weight: torch.Tensor) -> "BilinearQuadraticFactor":
        if weight.ndim != 2:
            raise ValueError("weight must be a matrix")
        model = cls(
            int(weight.shape[0]),
            int(weight.shape[1]),
            dtype=weight.dtype,
            device=weight.device,
        )
        with torch.no_grad():
            model.weight.copy_(weight)
        return model

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if x.shape != z.shape or x.shape[-1] != self.dimension:
            raise ValueError("x and z must have the same final dimension as the model")
        xw = x @ self.weight
        zw = z @ self.weight
        return (xw * zw).sum(dim=-1)

    def covariance(self) -> torch.Tensor:
        return self.weight @ self.weight.T

    def svd_contracted_copy(self, factors: int) -> "BilinearQuadraticFactor":
        """Target-free best rank-r covariance truncation with physical factor removal."""
        factors = int(factors)
        if not 1 <= factors <= min(self.dimension, self.factors):
            raise ValueError("invalid contracted factor count")
        with torch.no_grad():
            u, s, _ = torch.linalg.svd(self.weight, full_matrices=False)
            contracted_weight = u[:, :factors] * s[:factors]
        return BilinearQuadraticFactor.from_weight(contracted_weight)

    def clone(self) -> "BilinearQuadraticFactor":
        return copy.deepcopy(self)


def factor_forward_macs(dimension: int, factors: int) -> int:
    """Leading multiply-accumulate count for two projections plus final dot."""
    dimension = int(dimension)
    factors = int(factors)
    if dimension < 1 or factors < 1:
        raise ValueError("dimension and factors must be positive")
    return 2 * dimension * factors + factors


def population_matrix_risk(
    model: BilinearQuadraticFactor,
    a_star: torch.Tensor | np.ndarray,
) -> float:
    """Exact population half-MSE under independent isotropic x,z inputs."""
    if not torch.is_tensor(a_star):
        a_star = torch.as_tensor(
            a_star,
            dtype=model.weight.dtype,
            device=model.weight.device,
        )
    with torch.no_grad():
        delta = model.covariance() - a_star
        return 0.5 * float(torch.sum(delta * delta).detach().cpu())


def subspace_accessibility(
    model: BilinearQuadraticFactor,
    target_direction: torch.Tensor | np.ndarray,
    *,
    rtol: float = 1e-10,
) -> float:
    """Squared target projection onto the trainable factor column space."""
    w = model.weight.detach().cpu().numpy()
    u = np.asarray(
        target_direction.detach().cpu().numpy() if torch.is_tensor(target_direction) else target_direction,
        dtype=float,
    ).reshape(-1)
    u = u / np.linalg.norm(u)
    q, r = np.linalg.qr(w, mode="reduced")
    diag = np.abs(np.diag(r))
    if diag.size == 0:
        return 0.0
    rank = int(np.sum(diag > rtol * max(float(diag.max()), 1.0)))
    if rank == 0:
        return 0.0
    proj = q[:, :rank].T @ u
    return float(proj @ proj)


def top_eigenpair_overlap(
    model: BilinearQuadraticFactor,
    target_direction: torch.Tensor | np.ndarray,
) -> tuple[float, float, float]:
    """Top eigenvalue, squared teacher overlap, and second eigenvalue of W W'."""
    a = model.covariance().detach().cpu().numpy()
    u = np.asarray(
        target_direction.detach().cpu().numpy() if torch.is_tensor(target_direction) else target_direction,
        dtype=float,
    ).reshape(-1)
    u = u / np.linalg.norm(u)
    vals, vecs = np.linalg.eigh(a)
    top = float(vals[-1])
    second = float(vals[-2]) if len(vals) > 1 else 0.0
    overlap = float((u @ vecs[:, -1]) ** 2)
    return top, overlap, second
