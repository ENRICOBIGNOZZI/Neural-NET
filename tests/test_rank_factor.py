import copy
import math

import torch

from neural_net.rank_factor import (
    RankContractibleMLP,
    RankFactorLinear,
    deleted_scale_norm,
    physical_rank_parameter_count,
    rank_factor_group_gradient_energy,
)


def test_canonicalization_preserves_weight_and_fixes_component_gauge():
    torch.manual_seed(0)
    layer = RankFactorLinear(5, 4, 4)
    with torch.no_grad():
        layer.left[:, 0].mul_(3.0)
        layer.scale[0].div_(3.0)
        layer.right[1, :].mul_(0.2)
        layer.scale[1].div_(0.2)
    weight_before = layer.effective_weight().detach().clone()
    invariant_before = layer.component_frobenius_norms().detach().clone()

    layer.canonicalize_()

    assert torch.allclose(layer.effective_weight(), weight_before, atol=1e-7, rtol=1e-6)
    assert torch.allclose(layer.component_frobenius_norms(), invariant_before, atol=1e-7, rtol=1e-6)
    assert torch.allclose(torch.linalg.vector_norm(layer.left, dim=0), torch.ones(layer.rank), atol=1e-6)
    assert torch.allclose(torch.linalg.vector_norm(layer.right, dim=1), torch.ones(layer.rank), atol=1e-6)
    assert torch.allclose(layer.scale.abs(), invariant_before, atol=1e-6, rtol=1e-6)


def test_physical_rank_contraction_matches_zero_scale_embedding():
    torch.manual_seed(1)
    layer = RankFactorLinear(5, 4, 4)
    x = torch.randn(7, 5)
    keep = torch.tensor([0, 2])
    drop = torch.tensor([1, 3])

    ghost = layer.ghost_zeroed_copy(drop)
    compact = layer.contracted_copy(keep)

    assert torch.allclose(ghost(x), compact(x), atol=1e-7, rtol=1e-7)
    expected_jump = torch.linalg.vector_norm(layer.scale[drop]).item()
    assert math.isclose(deleted_scale_norm(layer, drop), expected_jump, rel_tol=1e-12)


def test_kept_gradients_match_between_ghost_and_compact_parameterizations():
    torch.manual_seed(2)
    layer = RankFactorLinear(3, 2, 4)
    keep = torch.tensor([0, 3])
    drop = torch.tensor([1, 2])
    x = torch.randn(8, 3)
    target = torch.randn(8, 2)

    ghost = layer.ghost_zeroed_copy(drop)
    compact = layer.contracted_copy(keep)

    ((ghost(x) - target) ** 2).mean().backward()
    ((compact(x) - target) ** 2).mean().backward()

    assert torch.allclose(ghost.left.grad[:, keep], compact.left.grad, atol=1e-7, rtol=1e-6)
    assert torch.allclose(ghost.scale.grad[keep], compact.scale.grad, atol=1e-7, rtol=1e-6)
    assert torch.allclose(ghost.right.grad[keep], compact.right.grad, atol=1e-7, rtol=1e-6)
    assert torch.allclose(ghost.bias.grad, compact.bias.grad, atol=1e-7, rtol=1e-6)


def test_group_gradient_energy_is_exact_parameter_group_norm():
    torch.manual_seed(3)
    layer = RankFactorLinear(4, 3, 5)
    x = torch.randn(6, 4)
    loss = layer(x).square().mean()
    loss.backward()

    energy = rank_factor_group_gradient_energy(layer)
    manual = []
    for q in range(layer.rank):
        manual.append(
            layer.left.grad[:, q].square().sum()
            + layer.scale.grad[q].square()
            + layer.right.grad[q, :].square().sum()
        )
    manual = torch.stack(manual)
    assert torch.allclose(energy, manual)


def test_rank_contraction_reduces_physical_parameter_count():
    layer = RankFactorLinear(10, 8, 6)
    compact = layer.contracted_copy([0, 2, 5])
    assert physical_rank_parameter_count(compact) < physical_rank_parameter_count(layer)
    expected_saved = 3 * (10 + 8 + 1)
    assert physical_rank_parameter_count(layer) - physical_rank_parameter_count(compact) == expected_saved


def test_rank_contractible_mlp_compact_copy_has_same_output_as_zeroed_rank_components():
    torch.manual_seed(4)
    model = RankContractibleMLP(input_dim=4, width1=6, width2=5, rank=4)
    keep = torch.tensor([0, 2])
    drop = torch.tensor([1, 3])
    x = torch.randn(9, 4)

    ghost = copy.deepcopy(model)
    with torch.no_grad():
        ghost.hidden2.scale[drop] = 0.0
    compact = model.contracted_rank_copy(keep)
    assert torch.allclose(ghost(x), compact(x), atol=1e-7, rtol=1e-7)
