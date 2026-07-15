import torch

from scripts.eval_verbal_report_causal import swap_lens_coordinates


def test_coordinate_swap_is_involution_and_preserves_orthogonal_component():
    torch.manual_seed(0)
    V = torch.randn(12, 2)
    h = torch.randn(5, 12)
    patched = swap_lens_coordinates(h, V)
    restored = swap_lens_coordinates(patched, V)
    assert torch.allclose(restored, h, atol=1e-5, rtol=1e-5)

    pinv = torch.linalg.pinv(V)
    orthogonal_delta = (patched - h) - ((patched - h) @ pinv.T) @ V.T
    assert torch.linalg.vector_norm(orthogonal_delta) < 1e-5


def test_coordinate_swap_handles_nonorthogonal_vectors():
    V = torch.tensor([[1.0, 1.0], [0.0, 1.0], [2.0, 0.0]])
    h = torch.tensor([[3.0, -2.0, 4.0]])
    twice = swap_lens_coordinates(swap_lens_coordinates(h, V), V)
    assert torch.allclose(twice, h, atol=1e-5, rtol=1e-5)
