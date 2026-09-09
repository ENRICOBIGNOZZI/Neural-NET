import torch

from neural_net.models import ContractibleMLP, num_parameters


def test_last_hidden_contraction_reduces_parameters():
    torch.manual_seed(0)
    model = ContractibleMLP(10, 16, 12)
    smaller = model.contracted_copy([0, 2, 4, 7, 9])
    assert smaller.width2 == 5
    assert num_parameters(smaller) < num_parameters(model)
    x = torch.randn(6, 10)
    assert smaller(x).shape == model(x).shape
