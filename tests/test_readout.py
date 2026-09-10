import torch

from neural_net.models import ContractibleMLP
from neural_net.readout import fit_ridge_readout


def test_ridge_readout_refit_does_not_increase_training_mse_for_small_alpha():
    torch.manual_seed(3)
    model = ContractibleMLP(5, 12, 8)
    x = torch.randn(80, 5)
    y = torch.randn(80)
    with torch.no_grad():
        before = torch.mean((model(x) - y) ** 2).item()
    fit_ridge_readout(model, x, y, alpha=1e-8)
    with torch.no_grad():
        after = torch.mean((model(x) - y) ** 2).item()
    assert after <= before + 1e-7
