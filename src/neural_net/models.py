from __future__ import annotations

import copy

import torch
from torch import nn


class ContractibleMLP(nn.Module):
    """Two-hidden-layer MLP whose last hidden layer can be physically contracted."""

    def __init__(self, input_dim: int, width1: int = 64, width2: int = 64):
        super().__init__()
        self.hidden1 = nn.Linear(input_dim, width1)
        self.hidden2 = nn.Linear(width1, width2)
        self.out = nn.Linear(width2, 1)
        self.act = nn.Tanh()

    @property
    def width2(self) -> int:
        return self.hidden2.out_features

    def features(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.act(self.hidden1(x))
        return self.act(self.hidden2(h1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.features(x)).squeeze(-1)

    def contracted_copy(self, keep_indices) -> "ContractibleMLP":
        """Return an exact neuron-subset copy at the contraction checkpoint.

        Retained neurons keep their incoming weights/biases and outgoing readout weights.
        The function changes because removed neurons are deleted; the controller is responsible
        for certifying that this current distortion is acceptably small.
        """
        idx = torch.as_tensor(keep_indices, dtype=torch.long, device=self.hidden2.weight.device)
        if idx.ndim != 1 or len(idx) == 0:
            raise ValueError("at least one neuron must be kept")
        new = ContractibleMLP(
            self.hidden1.in_features,
            self.hidden1.out_features,
            int(idx.numel()),
        ).to(self.hidden2.weight.device)
        new.hidden1.load_state_dict(copy.deepcopy(self.hidden1.state_dict()))
        with torch.no_grad():
            new.hidden2.weight.copy_(self.hidden2.weight[idx])
            new.hidden2.bias.copy_(self.hidden2.bias[idx])
            new.out.weight.copy_(self.out.weight[:, idx])
            new.out.bias.copy_(self.out.bias)
        return new


def num_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
