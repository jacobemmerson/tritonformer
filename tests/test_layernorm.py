import pytest
import torch
import torch.nn.functional as F
from model.kernels.layernorm import layernorm
from tests.conftest import TOLERANCES

TOL = TOLERANCES["layernorm"]


@pytest.mark.parametrize("shape", [
    (1, 64, 192),      # batch 1 -- grid edge
    (8, 64, 192),      # typical
    (512, 64, 192),    # large batch
    (4, 64, 192),
])
def test_matches_torch(device, shape):
    x = torch.randn(shape, device=device)
    w = torch.randn(shape[-1], device=device)
    b = torch.randn(shape[-1], device=device)
    expected = F.layer_norm(x, (shape[-1],), w, b, 1e-5)
    torch.testing.assert_close(layernorm(x, w, b, 1e-5), expected, **TOL)


def test_non_power_of_two_feature_dim(device):
    """D=192 is not a power of two -- the masked partial block is the most
    likely source of silent garbage in every kernel in this project."""
    x = torch.randn(4, 64, 192, device=device)
    w = torch.ones(192, device=device)
    b = torch.zeros(192, device=device)
    out = layernorm(x, w, b, 1e-5)
    torch.testing.assert_close(out, F.layer_norm(x, (192,), w, b, 1e-5), **TOL)
    torch.testing.assert_close(out.mean(-1),
                               torch.zeros(4, 64, device=device), atol=1e-5,
                               rtol=1e-4)


def test_non_contiguous_input(device):
    """Attention transposes before normalizing in some variants, so kernels
    receive strided tensors -- not freshly allocated contiguous ones."""
    base = torch.randn(4, 192, 64, device=device)
    x = base.transpose(1, 2)
    assert not x.is_contiguous()
    w = torch.randn(192, device=device)
    b = torch.randn(192, device=device)
    torch.testing.assert_close(layernorm(x, w, b, 1e-5),
                               F.layer_norm(x, (192,), w, b, 1e-5), **TOL)


def test_near_zero_variance_uses_eps(device):
    x = torch.full((4, 64, 192), 3.0, device=device)
    w = torch.ones(192, device=device)
    b = torch.zeros(192, device=device)
    out = layernorm(x, w, b, 1e-5)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, torch.zeros_like(out), atol=1e-4, rtol=0)
