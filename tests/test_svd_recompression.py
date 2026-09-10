import itertools

import torch

from neural_net.rank_factor import RankContractibleMLP, RankFactorLinear


def test_svd_contracted_copy_is_exact_truncated_svd_weight():
    torch.manual_seed(10)
    layer = RankFactorLinear(7, 6, 5).double()
    target_rank = 2
    weight = layer.effective_weight().detach()
    u, s, vh = torch.linalg.svd(weight, full_matrices=False)
    expected = (u[:, :target_rank] * s[:target_rank].unsqueeze(0)) @ vh[:target_rank, :]

    compact = layer.svd_contracted_copy(target_rank)

    assert compact.rank == target_rank
    assert torch.allclose(compact.effective_weight(), expected, atol=1e-11, rtol=1e-10)
    assert torch.allclose(compact.bias, layer.bias, atol=0.0, rtol=0.0)


def test_svd_recompression_beats_every_raw_component_subset_in_frobenius_error():
    torch.manual_seed(11)
    layer = RankFactorLinear(6, 5, 4).double()
    weight = layer.effective_weight().detach()
    target_rank = 2
    svd_compact = layer.svd_contracted_copy(target_rank)
    svd_error = torch.linalg.matrix_norm(weight - svd_compact.effective_weight(), ord="fro")

    subset_errors = []
    for keep in itertools.combinations(range(layer.rank), target_rank):
        candidate = layer.contracted_copy(keep)
        subset_errors.append(
            torch.linalg.matrix_norm(weight - candidate.effective_weight(), ord="fro")
        )

    assert svd_error <= torch.stack(subset_errors).min() + 1e-11


def test_singular_value_tail_fraction_equals_relative_truncated_svd_error():
    torch.manual_seed(12)
    layer = RankFactorLinear(8, 6, 5).double()
    target_rank = 3
    weight = layer.effective_weight().detach()
    compact = layer.svd_contracted_copy(target_rank)
    relative_error = (
        torch.linalg.matrix_norm(weight - compact.effective_weight(), ord="fro").square()
        / torch.linalg.matrix_norm(weight, ord="fro").square()
    )

    assert torch.allclose(
        layer.singular_value_tail_fraction(target_rank),
        relative_error,
        atol=1e-11,
        rtol=1e-10,
    )


def test_full_rank_svd_recompression_preserves_mlp_function():
    torch.manual_seed(13)
    model = RankContractibleMLP(input_dim=4, width1=6, width2=5, rank=3).double()
    x = torch.randn(19, 4, dtype=torch.float64)

    recompressed = model.svd_contracted_rank_copy(3)

    assert recompressed.rank == 3
    assert torch.allclose(model(x), recompressed(x), atol=1e-11, rtol=1e-10)
