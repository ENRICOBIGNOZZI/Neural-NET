import torch

from neural_net.quadratic_factor_net import (
    BilinearQuadraticFactor,
    factor_forward_macs,
)


def test_population_gradient_induces_riccati_covariance_velocity():
    torch.manual_seed(7)
    dimension = 5
    factors = 7
    weight = torch.randn(dimension, factors, dtype=torch.float64) / factors**0.5
    weight.requires_grad_(True)

    teacher_basis = torch.eye(dimension, dtype=torch.float64)[:, :2]
    a_star = teacher_basis @ teacher_basis.T
    covariance = weight @ weight.T
    risk = 0.5 * torch.sum((covariance - a_star) ** 2)
    gradient = torch.autograd.grad(risk, weight)[0]

    weight_velocity = -gradient
    covariance_velocity = weight_velocity @ weight.T + weight @ weight_velocity.T
    riccati = 2.0 * (a_star @ covariance + covariance @ a_star) - 4.0 * (
        covariance @ covariance
    )

    assert torch.allclose(covariance_velocity, riccati, atol=1e-10, rtol=1e-10)


def test_svd_contraction_is_top_covariance_truncation_and_removes_parameters():
    torch.manual_seed(11)
    model = BilinearQuadraticFactor(8, 12, dtype=torch.float64)
    reduced = model.svd_contracted_copy(3)

    with torch.no_grad():
        u, s, _ = torch.linalg.svd(model.weight, full_matrices=False)
        expected_covariance = (u[:, :3] * s[:3].square()) @ u[:, :3].T

    assert reduced.weight.shape == (8, 3)
    assert reduced.weight.numel() < model.weight.numel()
    assert torch.allclose(
        reduced.covariance(), expected_covariance, atol=1e-10, rtol=1e-10
    )


def test_confirmatory_factor_cost_ratio_is_exactly_four():
    wide = factor_forward_macs(32, 48)
    compact = factor_forward_macs(32, 12)
    assert wide / compact == 4.0
