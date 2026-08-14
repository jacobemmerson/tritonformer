import pytest
import torch
import torch.nn.functional as F
from model.kernels.softmax import softmax
from tests.conftest import TOLERANCES

TOL = TOLERANCES["softmax"]


@pytest.mark.parametrize("shape", [
    (1, 3, 64, 64), (8, 3, 64, 64), (512, 3, 64, 64), (4, 192),
])
def test_matches_torch(device, shape):
    x = torch.randn(shape, device=device)
    torch.testing.assert_close(softmax(x), F.softmax(x, dim=-1), **TOL)


def test_rows_sum_to_one(device):
    x = torch.randn(8, 3, 64, 64, device=device)
    sums = softmax(x).sum(-1)
    torch.testing.assert_close(sums, torch.ones_like(sums), **TOL)


def test_large_magnitude_inputs_do_not_overflow(device):
    """Without max-subtraction, exp(1e5) is inf and the row becomes NaN."""
    x = torch.full((4, 64), 1e4, device=device)
    x[:, 0] = 1e5
    out = softmax(x)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, F.softmax(x, dim=-1), **TOL)


def test_uniform_row_is_uniform(device):
    x = torch.zeros(4, 64, device=device)
    torch.testing.assert_close(softmax(x),
                               torch.full((4, 64), 1 / 64, device=device), **TOL)


def test_non_power_of_two_row(device):
    x = torch.randn(4, 192, device=device)
    torch.testing.assert_close(softmax(x), F.softmax(x, dim=-1), **TOL)
