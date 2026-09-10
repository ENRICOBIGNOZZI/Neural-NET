from __future__ import annotations

import copy

import torch
from torch import nn


class RankFactorLinear(nn.Module):
    """Linear map represented as a sum of trainable rank-one components.

    The weight matrix is

        W = left @ diag(scale) @ right.

    A component can be removed physically by deleting one column of ``left``, the
    matching scalar in ``scale``, and one row of ``right``.  In the full parameter
    space the same compact model has an exact ghost embedding obtained by setting
    the deleted scales to zero while keeping their left/right vectors fixed.

    The factors are structural rank-one components, not singular vectors.  The
    parameterization has a scale gauge; ``canonicalize_`` removes that gauge at a
    checkpoint by making the active left columns and right rows unit norm while
    absorbing their norms into ``scale`` without changing the represented matrix.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        *,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if in_features < 1 or out_features < 1 or rank < 1:
            raise ValueError("in_features, out_features, and rank must be positive")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = int(rank)

        self.left = nn.Parameter(torch.empty(self.out_features, self.rank))
        self.scale = nn.Parameter(torch.empty(self.rank))
        self.right = nn.Parameter(torch.empty(self.rank, self.in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.left, mean=0.0, std=self.out_features ** -0.5)
        nn.init.normal_(self.right, mean=0.0, std=self.in_features ** -0.5)
        nn.init.ones_(self.scale)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        self.canonicalize_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = x @ self.right.t()
        latent = latent * self.scale
        out = latent @ self.left.t()
        if self.bias is not None:
            out = out + self.bias
        return out

    def effective_weight(self) -> torch.Tensor:
        return (self.left * self.scale.unsqueeze(0)) @ self.right

    def component_frobenius_norms(self) -> torch.Tensor:
        """Gauge-invariant Frobenius norm of each represented rank-one term."""
        left_norm = torch.linalg.vector_norm(self.left, dim=0)
        right_norm = torch.linalg.vector_norm(self.right, dim=1)
        return self.scale.abs() * left_norm * right_norm

    @torch.no_grad()
    def canonicalize_(self, eps: float = 1e-12) -> "RankFactorLinear":
        """Fix the scale gauge without changing the represented affine map.

        For every nondegenerate component, absorb ``||left_q|| ||right_q||`` into
        ``scale_q`` and normalize the two direction factors.  Afterward
        ``abs(scale_q)`` equals the component Frobenius norm.  Degenerate components
        already represent zero and are assigned zero scale.
        """
        if eps <= 0:
            raise ValueError("eps must be positive")
        left_norm = torch.linalg.vector_norm(self.left, dim=0)
        right_norm = torch.linalg.vector_norm(self.right, dim=1)
        active = (left_norm > eps) & (right_norm > eps)
        multiplier = left_norm * right_norm

        if torch.any(active):
            self.scale[active].mul_(multiplier[active])
            self.left[:, active].div_(left_norm[active].unsqueeze(0))
            self.right[active, :].div_(right_norm[active].unsqueeze(1))
        if torch.any(~active):
            self.scale[~active].zero_()
        return self

    def contracted_copy(self, keep_indices) -> "RankFactorLinear":
        idx = torch.as_tensor(keep_indices, dtype=torch.long, device=self.scale.device)
        if idx.ndim != 1 or idx.numel() == 0:
            raise ValueError("at least one rank component must be kept")
        if torch.unique(idx).numel() != idx.numel():
            raise ValueError("keep_indices must be unique")
        if int(idx.min()) < 0 or int(idx.max()) >= self.rank:
            raise ValueError("keep index out of range")

        new = RankFactorLinear(
            self.in_features,
            self.out_features,
            int(idx.numel()),
            bias=self.bias is not None,
        ).to(device=self.scale.device, dtype=self.scale.dtype)
        with torch.no_grad():
            new.left.copy_(self.left[:, idx])
            new.scale.copy_(self.scale[idx])
            new.right.copy_(self.right[idx, :])
            if self.bias is not None:
                new.bias.copy_(self.bias)
        return new

    def ghost_zeroed_copy(self, drop_indices) -> "RankFactorLinear":
        """Same-size exact embedding of a physical contraction at the checkpoint."""
        idx = torch.as_tensor(drop_indices, dtype=torch.long, device=self.scale.device)
        new = copy.deepcopy(self)
        if idx.numel() == 0:
            return new
        if idx.ndim != 1 or torch.unique(idx).numel() != idx.numel():
            raise ValueError("drop_indices must be a unique one-dimensional index set")
        if int(idx.min()) < 0 or int(idx.max()) >= self.rank:
            raise ValueError("drop index out of range")
        with torch.no_grad():
            new.scale[idx] = 0.0
        return new


def deleted_scale_norm(layer: RankFactorLinear, drop_indices) -> float:
    """Parameter jump of the ghost embedding in the current gauge: ||scale_D||_2."""
    idx = torch.as_tensor(drop_indices, dtype=torch.long, device=layer.scale.device)
    if idx.numel() == 0:
        return 0.0
    return float(torch.linalg.vector_norm(layer.scale[idx]).detach().cpu())


def rank_factor_group_gradient_energy(layer: RankFactorLinear) -> torch.Tensor:
    """Exact squared gradient norm carried by each rank-one parameter group.

    Group q contains ``left[:, q]``, ``scale[q]``, and ``right[q, :]``.  The
    returned vector therefore sums to the gradient energy of all rank-factor
    parameters (excluding the shared bias).  Evaluate this after canonicalizing a
    checkpoint when a gauge-fixed structural diagnostic is desired.
    """
    if layer.left.grad is None or layer.scale.grad is None or layer.right.grad is None:
        raise ValueError("backward() must populate left, scale, and right gradients first")
    return (
        layer.left.grad.square().sum(dim=0)
        + layer.scale.grad.square()
        + layer.right.grad.square().sum(dim=1)
    )


def physical_rank_parameter_count(layer: RankFactorLinear) -> int:
    count = layer.rank * (layer.out_features + 1 + layer.in_features)
    if layer.bias is not None:
        count += layer.out_features
    return int(count)


class RankContractibleMLP(nn.Module):
    """Small MLP with a physically rank-contractible second hidden affine map."""

    def __init__(
        self,
        input_dim: int,
        width1: int = 64,
        width2: int = 64,
        rank: int = 32,
    ) -> None:
        super().__init__()
        self.hidden1 = nn.Linear(input_dim, width1)
        self.hidden2 = RankFactorLinear(width1, width2, rank)
        self.out = nn.Linear(width2, 1)
        self.act = nn.Tanh()

    @property
    def rank(self) -> int:
        return self.hidden2.rank

    def features(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.act(self.hidden1(x))
        return self.act(self.hidden2(h1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.features(x)).squeeze(-1)

    @torch.no_grad()
    def canonicalize_rank_components_(self) -> "RankContractibleMLP":
        self.hidden2.canonicalize_()
        return self

    def contracted_rank_copy(self, keep_indices) -> "RankContractibleMLP":
        idx = torch.as_tensor(keep_indices, dtype=torch.long, device=self.hidden2.scale.device)
        if idx.ndim != 1 or idx.numel() == 0:
            raise ValueError("at least one rank component must be kept")
        new = RankContractibleMLP(
            self.hidden1.in_features,
            self.hidden1.out_features,
            self.hidden2.out_features,
            int(idx.numel()),
        ).to(device=self.hidden2.scale.device, dtype=self.hidden2.scale.dtype)
        new.hidden1.load_state_dict(copy.deepcopy(self.hidden1.state_dict()))
        new.hidden2 = self.hidden2.contracted_copy(idx)
        new.out.load_state_dict(copy.deepcopy(self.out.state_dict()))
        return new
