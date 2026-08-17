import pytest
import torch
from model.baseline.layers import block as block_reference
from model.config import ViTConfig
from model.kernels.block import block_composed, block_fused
from tests.conftest import TOLERANCES

TOL = TOLERANCES["block"]
VARIANTS = [block_composed, block_fused]


def params(device, cfg=ViTConfig()):
    dim, hidden = cfg.dim, cfg.mlp_hidden
    scaled = lambda *shape: torch.randn(*shape, device=device) * 0.05
    return dict(
        ln1_w=torch.ones(dim, device=device),
        ln1_b=torch.zeros(dim, device=device),
        qkv_w=scaled(3 * dim, dim), qkv_b=torch.zeros(3 * dim, device=device),
        proj_w=scaled(dim, dim), proj_b=torch.zeros(dim, device=device),
        ln2_w=torch.ones(dim, device=device),
        ln2_b=torch.zeros(dim, device=device),
        w1=scaled(hidden, dim), b1=torch.zeros(hidden, device=device),
        w2=scaled(dim, hidden), b2=torch.zeros(dim, device=device),
        heads=cfg.heads, scale=cfg.scale, eps=cfg.eps)


@pytest.mark.parametrize("fn", VARIANTS)
@pytest.mark.parametrize("batch", [1, 8, 128])
def test_matches_reference(device, fn, batch):
    x = torch.randn(batch, 64, 192, device=device)
    p = params(device)
    torch.testing.assert_close(fn(x, **p), block_reference(x, **p), **TOL)


@pytest.mark.parametrize("fn", VARIANTS)
def test_residual_path_preserved(device, fn):
    """With proj_w left non-zero and only w2 zeroed, the MLP contributes
    nothing but the attention residual (proj_out + x) still carries a
    real signal, so the block's output must equal that post-attention
    residual -- not x.

    An earlier version of this test zeroed proj_w as well as w2. That
    made `residual = proj_out + x` collapse to exactly `x` for both the
    correct wiring and a buggy one that applies the second residual to
    the pre-attention `x` instead of the post-attention `residual`: with
    proj_w zeroed, those two quantities are identical, so the mis-wired
    variant reproduced the correct output and the test could not
    discriminate. Keeping proj_w non-zero makes the two paths diverge:
    the correct block returns `residual = proj_out + x`, the mis-wired
    one would return `x` again, so asserting output != x is exactly the
    check that catches a residual applied to the wrong tensor.
    """
    x = torch.randn(4, 64, 192, device=device)
    p = params(device)
    p["w2"] = torch.zeros_like(p["w2"])
    out = fn(x, **p)
    torch.testing.assert_close(out, block_reference(x, **p), **TOL)
    # A mis-wired second residual (applied to x instead of the
    # post-attention residual) would return exactly x here, since the
    # MLP contributes nothing. Requiring a real gap catches that bug.
    max_diff = (out - x).abs().max().item()
    assert max_diff > 1e-3, (
        f"block output is indistinguishable from x (max abs diff "
        f"{max_diff}); the post-attention residual never propagated")
