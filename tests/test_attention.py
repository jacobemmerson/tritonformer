import pytest
import torch
import torch.nn.functional as F
from model.kernels.attention import (attention_composed, attention_composed_bmm,
                                     attention_flash, attention_flash_tiled, bmm)
from tests.conftest import TOLERANCES

TOL = TOLERANCES["attention"]
VARIANTS = [attention_composed, attention_composed_bmm,
            attention_flash, attention_flash_tiled]
FLASH_VARIANTS = [attention_flash, attention_flash_tiled]


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


@pytest.mark.parametrize("fn", FLASH_VARIANTS)
def test_flash_never_materializes_scores(device, fn):
    """A [B,H,S,S] score buffer for batch 512 would be 400MB. If the kernel
    allocates one, this OOMs or shows a large allocation spike.

    At S = head_dim = 64, out_bytes == score_bytes numerically, so a
    threshold of `< score_bytes` is unsatisfiable by construction: the
    kernel's own mandatory output allocation already equals score_bytes,
    before any score buffer is even considered. The threshold must
    account for the output allocation the kernel is required to make;
    it should only fail an implementation that allocates a score buffer
    on top of that.
    """
    torch.cuda.reset_peak_memory_stats()
    q, k, v = (torch.randn(256, 3, 64, 64, device=device) for _ in range(3))
    baseline = torch.cuda.max_memory_allocated()
    fn(q, k, v, 64 ** -0.5)
    peak = torch.cuda.max_memory_allocated()
    out_bytes = 256 * 3 * 64 * 64 * 4      # the output the kernel must allocate
    score_bytes = 256 * 3 * 64 * 64 * 4    # the [B,H,S,S] buffer it must NOT allocate
    assert peak - baseline < out_bytes + score_bytes


def test_bmm_matches_torch_matmul_with_scale(device):
    a = torch.randn(2, 3, 64, 64, device=device)
    b = torch.randn(2, 3, 64, 64, device=device)
    expected = torch.matmul(a, b.transpose(-2, -1)) * 0.125
    torch.testing.assert_close(bmm(a, b.transpose(-2, -1), 0.125), expected, **TOL)


def test_bmm_handles_ragged_shapes(device):
    """None of M, K, N is a multiple of the tile, so every mask is exercised."""
    a = torch.randn(2, 3, 40, 24, device=device)
    b = torch.randn(2, 3, 24, 72, device=device)
    torch.testing.assert_close(bmm(a, b), torch.matmul(a, b), **TOL)


def test_bmm_reads_transposed_view_without_copy(device):
    """q @ k.T must read k through its strides. A hidden .contiguous() on the
    transposed view would add a [B, H, D, S] copy, which is exactly the kind of
    traffic this arm exists to keep out of the comparison."""
    a = torch.randn(64, 3, 64, 64, device=device)
    b = torch.randn(64, 3, 64, 64, device=device)
    bmm(a, b.transpose(-2, -1))
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    bmm(a, b.transpose(-2, -1))
    out_bytes = 64 * 3 * 64 * 64 * 4
    assert torch.cuda.max_memory_allocated() - baseline <= out_bytes


@pytest.mark.parametrize("fn", [attention_composed_bmm, attention_flash_tiled])
def test_partial_key_tile_matches_sdpa(device, fn):
    """seq 40 leaves the second 32-wide key tile half masked and the query
    grid with a ragged last block."""
    q, k, v = (torch.randn(2, 3, 40, 64, device=device) for _ in range(3))
    expected = F.scaled_dot_product_attention(q, k, v)
    torch.testing.assert_close(fn(q, k, v, 64 ** -0.5), expected, **TOL)


def test_tiled_with_one_key_tile_matches_single_tile_kernel(device):
    """With block_n covering the whole sequence the online softmax degenerates
    to the old kernel's single pass, so the two should agree far more tightly
    than either agrees with SDPA."""
    q, k, v = (torch.randn(4, 3, 64, 64, device=device) for _ in range(3))
    tiled = attention_flash_tiled(q, k, v, 64 ** -0.5, block_m=64, block_n=64)
    torch.testing.assert_close(tiled, attention_flash(q, k, v, 64 ** -0.5),
                               rtol=1e-5, atol=1e-5)
