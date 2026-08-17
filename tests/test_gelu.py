import pytest
import torch
import torch.nn.functional as F
from model.kernels.gelu import gelu
from tests.conftest import TOLERANCES

TOL = TOLERANCES["gelu"]


@pytest.mark.parametrize("shape", [(1, 64, 768), (8, 64, 768), (512, 64, 768)])
def test_matches_torch(device, shape):
    x = torch.randn(shape, device=device)
    torch.testing.assert_close(gelu(x), F.gelu(x, approximate="tanh"), **TOL)


def test_non_contiguous_input(device):
    x = torch.randn(4, 768, 64, device=device).transpose(1, 2)
    assert not x.is_contiguous()
    torch.testing.assert_close(gelu(x), F.gelu(x, approximate="tanh"), **TOL)


def test_saturates_without_overflow(device):
    x = torch.tensor([-30.0, -10.0, 0.0, 10.0, 30.0], device=device)
    out = gelu(x)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, F.gelu(x, approximate="tanh"), **TOL)


def test_odd_element_count(device):
    x = torch.randn(1023, device=device)
    torch.testing.assert_close(gelu(x), F.gelu(x, approximate="tanh"), **TOL)
