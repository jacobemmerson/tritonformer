import torch
import torch.nn.functional as F
from model.kernels.attention import qkv_project
from tests.conftest import TOLERANCES

TOL = TOLERANCES["linear"]


def test_matches_three_separate_projections(device):
    x = torch.randn(8, 64, 192, device=device)
    qkv_w = torch.randn(576, 192, device=device) * 0.05
    qkv_b = torch.randn(576, device=device)

    packed = F.linear(x, qkv_w, qkv_b)
    expected = packed.reshape(8, 64, 3, 3, 64).permute(2, 0, 3, 1, 4)

    q, k, v = qkv_project(x, qkv_w, qkv_b, heads=3)
    torch.testing.assert_close(q, expected[0], **TOL)
    torch.testing.assert_close(k, expected[1], **TOL)
    torch.testing.assert_close(v, expected[2], **TOL)


def test_shapes(device):
    x = torch.randn(4, 64, 192, device=device)
    qkv_w = torch.randn(576, 192, device=device) * 0.05
    qkv_b = torch.zeros(576, device=device)
    q, k, v = qkv_project(x, qkv_w, qkv_b, heads=3)
    for tensor in (q, k, v):
        assert tensor.shape == (4, 3, 64, 64)


def test_batch_one(device):
    x = torch.randn(1, 64, 192, device=device)
    qkv_w = torch.randn(576, 192, device=device) * 0.05
    qkv_b = torch.randn(576, device=device)
    q, k, v = qkv_project(x, qkv_w, qkv_b, heads=3)
    packed = F.linear(x, qkv_w, qkv_b).reshape(1, 64, 3, 3, 64)
    torch.testing.assert_close(q, packed.permute(2, 0, 3, 1, 4)[0], **TOL)


def test_ordering_with_heads_not_equal_to_three(device):
    """heads=4 (not 3) so the QKV axis (size 3) and the heads axis (size 4)
    differ in size. With heads=3, both axes coincidentally have size 3, so a
    reshape/permute that swapped them would still produce the right *shape*
    and only be caught because the compared values happen to differ. Using
    heads=4 makes the two axes structurally distinguishable, so this test's
    discriminating power does not depend on that coincidence."""
    x = torch.randn(4, 64, 192, device=device)
    qkv_w = torch.randn(576, 192, device=device) * 0.05
    qkv_b = torch.randn(576, device=device)

    expected = F.linear(x, qkv_w, qkv_b).reshape(4, 64, 3, 4, 48).permute(2, 0, 3, 1, 4)

    q, k, v = qkv_project(x, qkv_w, qkv_b, heads=4)
    for tensor in (q, k, v):
        assert tensor.shape == (4, 4, 64, 48)
    torch.testing.assert_close(q, expected[0], **TOL)
    torch.testing.assert_close(k, expected[1], **TOL)
    torch.testing.assert_close(v, expected[2], **TOL)
