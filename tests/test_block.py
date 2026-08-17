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
    """With zeroed output projections the block must be close to identity,
    which catches a dropped or double-applied residual."""
    x = torch.randn(4, 64, 192, device=device)
    p = params(device)
    p["proj_w"] = torch.zeros_like(p["proj_w"])
    p["w2"] = torch.zeros_like(p["w2"])
    torch.testing.assert_close(fn(x, **p), x, **TOL)
