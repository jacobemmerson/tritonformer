import torch
import torch.nn.functional as F
from model.kernels.layernorm import layernorm_residual
from tests.conftest import TOLERANCES

TOL = TOLERANCES["layernorm"]


def test_matches_unfused_sequence(device):
    x = torch.randn(8, 64, 192, device=device)
    residual = torch.randn(8, 64, 192, device=device)
    w = torch.randn(192, device=device)
    b = torch.randn(192, device=device)

    expected_residual = x + residual
    expected_normed = F.layer_norm(expected_residual, (192,), w, b, 1e-5)

    normed, updated = layernorm_residual(x, residual, w, b, 1e-5)
    torch.testing.assert_close(updated, expected_residual, **TOL)
    torch.testing.assert_close(normed, expected_normed, **TOL)


def test_batch_one(device):
    x = torch.randn(1, 64, 192, device=device)
    residual = torch.randn(1, 64, 192, device=device)
    w = torch.ones(192, device=device)
    b = torch.zeros(192, device=device)
    normed, updated = layernorm_residual(x, residual, w, b, 1e-5)
    torch.testing.assert_close(updated, x + residual, **TOL)
    torch.testing.assert_close(
        normed, F.layer_norm(x + residual, (192,), w, b, 1e-5), **TOL)


def test_residual_is_written_not_aliased(device):
    x = torch.randn(4, 64, 192, device=device)
    residual = torch.zeros(4, 64, 192, device=device)
    w = torch.ones(192, device=device)
    b = torch.zeros(192, device=device)
    _, updated = layernorm_residual(x, residual, w, b, 1e-5)
    assert updated.data_ptr() != residual.data_ptr()
    torch.testing.assert_close(updated, x, **TOL)
