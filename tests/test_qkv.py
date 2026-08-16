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
