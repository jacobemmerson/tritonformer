import pytest
import torch
import torch.nn.functional as F
from model.kernels.linear import linear
from tests.conftest import TOLERANCES

TOL = TOLERANCES["linear"]

# (in_features, out_features) actually used by the model.
SHAPES = [(192, 576), (192, 192), (192, 768), (768, 192)]


@pytest.mark.parametrize("batch", [1, 8, 128])
@pytest.mark.parametrize("k,n", SHAPES)
def test_matches_torch(device, batch, k, n):
    x = torch.randn(batch, 64, k, device=device)
    w = torch.randn(n, k, device=device) * 0.05
    b = torch.randn(n, device=device)
    torch.testing.assert_close(linear(x, w, b), F.linear(x, w, b), **TOL)


def test_without_bias(device):
    x = torch.randn(4, 64, 192, device=device)
    w = torch.randn(192, 192, device=device) * 0.05
    torch.testing.assert_close(linear(x, w, None), F.linear(x, w, None), **TOL)


def test_non_contiguous_input(device):
    x = torch.randn(4, 192, 64, device=device).transpose(1, 2)
    assert not x.is_contiguous()
    w = torch.randn(192, 192, device=device) * 0.05
    b = torch.randn(192, device=device)
    torch.testing.assert_close(linear(x, w, b), F.linear(x, w, b), **TOL)


def test_non_power_of_two_k_dimension(device):
    """K=100 with BLOCK_K=32 is 3 full tiles plus a 4-element remainder, a
    genuine partial tile on the reduction axis. (K=192 divides evenly into
    6 full BLOCK_K=32 tiles and would never exercise the K mask at all.)
    An unmasked load on that remainder silently adds garbage to the
    accumulator."""
    x = torch.randn(2, 64, 100, device=device)
    w = torch.eye(100, device=device)
    out = linear(x, w, None)
    torch.testing.assert_close(out, x, **TOL)
