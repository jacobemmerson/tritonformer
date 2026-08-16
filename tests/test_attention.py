import pytest
import torch
import torch.nn.functional as F
from model.kernels.attention import attention_composed
from tests.conftest import TOLERANCES

TOL = TOLERANCES["attention"]
VARIANTS = [attention_composed]


@pytest.mark.parametrize("fn", VARIANTS)
@pytest.mark.parametrize("batch", [1, 8, 128])
def test_matches_sdpa(device, fn, batch):
    q, k, v = (torch.randn(batch, 3, 64, 64, device=device) for _ in range(3))
    expected = F.scaled_dot_product_attention(q, k, v)
    torch.testing.assert_close(fn(q, k, v, 64 ** -0.5), expected, **TOL)


@pytest.mark.parametrize("fn", VARIANTS)
def test_output_is_convex_combination_of_v(device, fn):
    """With identical queries and keys the attention is uniform, so the
    output must be the mean of V along the sequence axis."""
    q = torch.zeros(2, 3, 64, 64, device=device)
    k = torch.zeros(2, 3, 64, 64, device=device)
    v = torch.randn(2, 3, 64, 64, device=device)
    out = fn(q, k, v, 64 ** -0.5)
    expected = v.mean(dim=2, keepdim=True).expand_as(v)
    torch.testing.assert_close(out, expected, **TOL)


@pytest.mark.parametrize("fn", VARIANTS)
def test_large_scores_do_not_overflow(device, fn):
    q = torch.full((2, 3, 64, 64), 50.0, device=device)
    k = torch.full((2, 3, 64, 64), 50.0, device=device)
    v = torch.randn(2, 3, 64, 64, device=device)
    out = fn(q, k, v, 64 ** -0.5)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("fn", VARIANTS)
def test_non_contiguous_head_split_layout(device, fn):
    """Real callers reach attention after reshape+permute, so q/k/v are
    views into one packed QKV buffer, not contiguous allocations."""
    qkv = torch.randn(4, 64, 3, 3, 64, device=device)
    qkv = qkv.permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    assert not q.is_contiguous()
    expected = F.scaled_dot_product_attention(q, k, v)
    torch.testing.assert_close(fn(q, k, v, 64 ** -0.5), expected, **TOL)
