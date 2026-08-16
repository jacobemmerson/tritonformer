import pytest
import torch
import torch.nn.functional as F
from model.kernels.mlp import mlp_composed, mlp_fused
from tests.conftest import TOLERANCES

TOL = TOLERANCES["mlp"]
VARIANTS = [mlp_composed, mlp_fused]


@pytest.mark.parametrize("fn", VARIANTS)
@pytest.mark.parametrize("batch", [1, 8, 128])
def test_matches_torch(device, fn, batch):
    x = torch.randn(batch, 64, 192, device=device)
    w1 = torch.randn(768, 192, device=device) * 0.05
    b1 = torch.randn(768, device=device)
    w2 = torch.randn(192, 768, device=device) * 0.05
    b2 = torch.randn(192, device=device)
    expected = F.linear(F.gelu(F.linear(x, w1, b1), approximate="tanh"), w2, b2)
    torch.testing.assert_close(fn(x, w1, b1, w2, b2), expected, **TOL)


@pytest.mark.parametrize("fn", VARIANTS)
def test_preserves_shape(device, fn):
    x = torch.randn(4, 64, 192, device=device)
    w1 = torch.randn(768, 192, device=device) * 0.05
    b1 = torch.zeros(768, device=device)
    w2 = torch.randn(192, 768, device=device) * 0.05
    b2 = torch.zeros(192, device=device)
    assert fn(x, w1, b1, w2, b2).shape == x.shape


def test_fused_matches_composed_exactly_enough(device):
    """Both are Triton and share accumulation strategy, so they should agree
    more tightly than either agrees with torch."""
    x = torch.randn(8, 64, 192, device=device)
    w1 = torch.randn(768, 192, device=device) * 0.05
    b1 = torch.randn(768, device=device)
    w2 = torch.randn(192, 768, device=device) * 0.05
    b2 = torch.randn(192, device=device)
    torch.testing.assert_close(mlp_fused(x, w1, b1, w2, b2),
                               mlp_composed(x, w1, b1, w2, b2),
                               rtol=1e-5, atol=1e-5)
