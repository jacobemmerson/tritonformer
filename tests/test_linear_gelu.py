import pytest
import torch
import torch.nn.functional as F
from model.kernels.linear import linear_gelu, linear_gelu_tuned
from tests.conftest import TOLERANCES

TOL = TOLERANCES["linear"]
VARIANTS = [linear_gelu, linear_gelu_tuned]


@pytest.mark.parametrize("fn", VARIANTS)
@pytest.mark.parametrize("batch", [1, 8, 128])
def test_matches_unfused_sequence(device, fn, batch):
    x = torch.randn(batch, 64, 192, device=device)
    w = torch.randn(768, 192, device=device) * 0.05
    b = torch.randn(768, device=device)
    expected = F.gelu(F.linear(x, w, b), approximate="tanh")
    torch.testing.assert_close(fn(x, w, b), expected, **TOL)


@pytest.mark.parametrize("fn", VARIANTS)
def test_second_mlp_shape(device, fn):
    x = torch.randn(4, 64, 768, device=device)
    w = torch.randn(192, 768, device=device) * 0.05
    b = torch.randn(192, device=device)
    expected = F.gelu(F.linear(x, w, b), approximate="tanh")
    torch.testing.assert_close(fn(x, w, b), expected, **TOL)


@pytest.mark.parametrize("fn", VARIANTS)
def test_large_activations_do_not_overflow(device, fn):
    x = torch.full((2, 64, 192), 10.0, device=device)
    w = torch.full((768, 192), 0.5, device=device)
    b = torch.zeros(768, device=device)
    out = fn(x, w, b)
    assert torch.isfinite(out).all()
    expected = F.gelu(F.linear(x, w, b), approximate="tanh")
    torch.testing.assert_close(out, expected, **TOL)
