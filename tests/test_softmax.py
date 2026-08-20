import pytest
import torch
import torch.nn.functional as F
from model.kernels.softmax import softmax, softmax_tuned
from tests.conftest import TOLERANCES

TOL = TOLERANCES["softmax"]
VARIANTS = [softmax, softmax_tuned]


@pytest.mark.parametrize("fn", VARIANTS)
@pytest.mark.parametrize("shape", [
    (1, 3, 64, 64), (8, 3, 64, 64), (512, 3, 64, 64), (4, 192),
])
def test_matches_torch(device, fn, shape):
    x = torch.randn(shape, device=device)
    torch.testing.assert_close(fn(x), F.softmax(x, dim=-1), **TOL)


@pytest.mark.parametrize("fn", VARIANTS)
def test_rows_sum_to_one(device, fn):
    x = torch.randn(8, 3, 64, 64, device=device)
    sums = fn(x).sum(-1)
    torch.testing.assert_close(sums, torch.ones_like(sums), **TOL)


@pytest.mark.parametrize("fn", VARIANTS)
def test_large_magnitude_inputs_do_not_overflow(device, fn):
    """Without max-subtraction, exp(1e5) is inf and the row becomes NaN."""
    x = torch.full((4, 64), 1e4, device=device)
    x[:, 0] = 1e5
    out = fn(x)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, F.softmax(x, dim=-1), **TOL)


@pytest.mark.parametrize("fn", VARIANTS)
def test_uniform_row_is_uniform(device, fn):
    x = torch.zeros(4, 64, device=device)
    torch.testing.assert_close(fn(x),
                               torch.full((4, 64), 1 / 64, device=device), **TOL)


@pytest.mark.parametrize("fn", VARIANTS)
def test_non_power_of_two_row(device, fn):
    x = torch.randn(4, 192, device=device)
    torch.testing.assert_close(fn(x), F.softmax(x, dim=-1), **TOL)
