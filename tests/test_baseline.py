import torch
import torch.nn.functional as F
from model.baseline.layers import attention, block, gelu, layernorm, linear, mlp, softmax
from tests.conftest import TOLERANCES


def test_layernorm_matches_torch(device):
    x = torch.randn(8, 64, 192, device=device)
    w = torch.randn(192, device=device)
    b = torch.randn(192, device=device)
    expected = F.layer_norm(x, (192,), w, b, 1e-5)
    torch.testing.assert_close(layernorm(x, w, b, 1e-5), expected,
                               **TOLERANCES["layernorm"])


def test_gelu_matches_torch(device):
    x = torch.randn(8, 64, 192, device=device)
    expected = F.gelu(x, approximate="tanh")
    torch.testing.assert_close(gelu(x), expected, **TOLERANCES["gelu"])


def test_softmax_over_last_dim(device):
    x = torch.randn(8, 3, 64, 64, device=device)
    torch.testing.assert_close(softmax(x), F.softmax(x, dim=-1),
                               **TOLERANCES["softmax"])


def test_softmax_is_numerically_stable(device):
    x = torch.full((4, 64), 1e4, device=device)
    x[:, 0] = 1e5
    out = softmax(x)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out.sum(-1), torch.ones(4, device=device),
                               **TOLERANCES["softmax"])


def test_linear_matches_torch(device):
    x = torch.randn(8, 64, 192, device=device)
    w = torch.randn(576, 192, device=device)
    b = torch.randn(576, device=device)
    torch.testing.assert_close(linear(x, w, b), F.linear(x, w, b),
                               **TOLERANCES["linear"])


def test_attention_matches_sdpa(device):
    q, k, v = (torch.randn(4, 3, 64, 64, device=device) for _ in range(3))
    expected = F.scaled_dot_product_attention(q, k, v)
    torch.testing.assert_close(attention(q, k, v, 64 ** -0.5), expected,
                               **TOLERANCES["attention"])


def test_mlp_shape_and_value(device):
    x = torch.randn(4, 64, 192, device=device)
    w1 = torch.randn(768, 192, device=device)
    b1 = torch.randn(768, device=device)
    w2 = torch.randn(192, 768, device=device)
    b2 = torch.randn(192, device=device)
    expected = F.linear(F.gelu(F.linear(x, w1, b1), approximate="tanh"), w2, b2)
    out = mlp(x, w1, b1, w2, b2)
    assert out.shape == x.shape
    torch.testing.assert_close(out, expected, **TOLERANCES["mlp"])


def test_block_matches_independent_prenorm_composition(device):
    """block is the oracle for every Triton block variant AND for the
    end-to-end gate, which compare it only against itself. Verify it against
    an independently written pre-norm composition instead."""
    torch.manual_seed(0)
    batch, seq, dim, heads, hidden = 4, 64, 192, 3, 768
    head_dim = dim // heads
    x = torch.randn(batch, seq, dim, device=device)
    p = dict(
        ln1_w=torch.randn(dim, device=device), ln1_b=torch.randn(dim, device=device),
        qkv_w=torch.randn(3 * dim, dim, device=device) * 0.05,
        qkv_b=torch.randn(3 * dim, device=device),
        proj_w=torch.randn(dim, dim, device=device) * 0.05,
        proj_b=torch.randn(dim, device=device),
        ln2_w=torch.randn(dim, device=device), ln2_b=torch.randn(dim, device=device),
        w1=torch.randn(hidden, dim, device=device) * 0.05,
        b1=torch.randn(hidden, device=device),
        w2=torch.randn(dim, hidden, device=device) * 0.05,
        b2=torch.randn(dim, device=device),
    )

    normed = F.layer_norm(x, (dim,), p["ln1_w"], p["ln1_b"], 1e-5)
    qkv = F.linear(normed, p["qkv_w"], p["qkv_b"])
    qkv = qkv.reshape(batch, seq, 3, heads, head_dim).permute(2, 0, 3, 1, 4)
    attended = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2])
    attended = attended.transpose(1, 2).reshape(batch, seq, dim)
    after_attn = x + F.linear(attended, p["proj_w"], p["proj_b"])
    normed2 = F.layer_norm(after_attn, (dim,), p["ln2_w"], p["ln2_b"], 1e-5)
    expected = after_attn + F.linear(
        F.gelu(F.linear(normed2, p["w1"], p["b1"]), approximate="tanh"),
        p["w2"], p["b2"])

    out = block(x, **p, heads=heads, scale=head_dim ** -0.5, eps=1e-5)
    assert out.shape == x.shape
    torch.testing.assert_close(out, expected, **TOLERANCES["block"])
