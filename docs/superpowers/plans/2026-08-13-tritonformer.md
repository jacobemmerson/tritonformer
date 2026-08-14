# Tritonformer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the forward pass of a shallow vision transformer entirely in Triton kernels, and measure with Nsight when kernel fusion helps performance and when it hurts.

**Architecture:** A pure-PyTorch reference implementation acts as both numerical oracle and performance baseline. A variant registry maps `(component, variant)` pairs to callables, so Triton kernels are swapped in one at a time against an always-working model. Each component then climbs a fusion ladder from unfused through progressively fused variants, with latency measured by `do_bench` and hardware counters collected by `ncu`.

**Tech Stack:** Python 3.10+, PyTorch 2.x (CUDA), Triton 2.2+, pytest, NVIDIA Nsight Compute (`ncu`) and Nsight Systems (`nsys`).

## Global Constraints

- **Forward pass only.** No backward passes, training loops, or optimizers in Triton. The reference model is trained in stock PyTorch.
- **Model config:** patch 4×4 → 64 tokens, mean pooling (no CLS token), dim 192, depth 6, 3 heads × head_dim 64, MLP hidden 768, pre-norm.
- **Primary dtype fp32.** fp16 is a later experiment, never a correctness fallback.
- **Batch sweep axis:** `{1, 8, 32, 128, 512}`, truncated to whatever fits in 4GB VRAM.
- **Target hardware:** GTX 1650 Ti (TU117, sm_75, 16 SMs, ~192 GB/s, 64KB shared memory/SM, **no tensor cores**). Secondary: Modal L4 or A10G.
- **Latency comes from `do_bench`. Counters come from `ncu`.** These are written to separate CSV files and never mixed.
- **Tolerances are declared per kernel and never loosened to make a test pass.** A test needing a looser tolerance is a bug report.
- **The reference implementation stays boring.** Plain `torch.nn.functional`, no optimization.
- **Every benchmark row records `commit_sha`, `sm_clock_mhz`, `temp_c`, and a `flagged` column.**

## File Structure

```
model/
  baseline/           pure PyTorch reference — oracle and baseline
    layers.py         layernorm, gelu, softmax, linear, attention, mlp, block
    vit.py            VisionTransformer assembled from baseline layers
  kernels/            Triton kernels; unfused and fused variants live side by side
    layernorm.py      layernorm, layernorm_residual
    gelu.py           gelu
    softmax.py        softmax
    linear.py         linear, linear_gelu
    attention.py      attention_composed, attention_qkv_fused, attention_flash
    mlp.py            mlp_fused
    block.py          block_fused
  registry.py         Component enum, register/get, VariantConfig
  config.py           ViTConfig dataclass
  vit.py              backend-switchable VisionTransformer
bench/
  harness.py          interleaved do_bench -> latency CSV
  profile.py          ncu wrapper -> counter CSV
  clocks.py           clock locking + telemetry sampling
  results/            *.csv
tests/
  conftest.py         shared fixtures, tolerance table
  test_registry.py
  test_layernorm.py   ... one per component
  test_end_to_end.py
scripts/
  probe_hardware.py   step 0 capability probe
  train_reference.py  trains and freezes the checkpoint
  lock_clocks.sh
docs/findings/        per-kernel markdown interpretation
data/                 CIFAR-10 + frozen checkpoint
```

**Note on existing scaffold:** the repo already contains empty `model/baseline/`, `model/fused/`, and `data/utils/`. This plan uses `model/baseline/` as designed and introduces `model/kernels/`. `model/fused/` is left in place — fused and unfused Triton variants deliberately live in the same files, because the registry is what distinguishes them and splitting by fusion level would fight that design. Removing `model/fused/` is recommended but requires your approval.

---

### Task 1: Hardware capability probe

Resolves the single largest unknown — whether `tl.dot` works on a tensor-core-less sm_75 card — before any design depends on the answer.

**Files:**
- Create: `scripts/probe_hardware.py`

**Interfaces:**
- Consumes: nothing
- Produces: `probe()` writing a human-readable report to stdout; no importable API.

- [ ] **Step 1: Write the probe script**

```python
"""Step 0: determine whether tl.dot is usable on this GPU.

Run this before any kernel work. If tl.dot fails or is catastrophically
slow here, matmul and attention work moves to Modal.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                   stride_am, stride_ak, stride_bk, stride_bn,
                   stride_cm, stride_cn,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                   BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=mask)


def try_dot(dtype, M=512, N=512, K=512):
    a = torch.randn((M, K), device="cuda", dtype=dtype)
    b = torch.randn((K, N), device="cuda", dtype=dtype)
    c = torch.empty((M, N), device="cuda", dtype=dtype)
    grid = (triton.cdiv(M, 64), triton.cdiv(N, 64))
    _matmul_kernel[grid](a, b, c, M, N, K,
                         a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                         c.stride(0), c.stride(1),
                         BLOCK_M=64, BLOCK_N=64, BLOCK_K=32)
    ref = (a.float() @ b.float()).to(dtype)
    max_err = (c.float() - ref.float()).abs().max().item()
    ms = triton.testing.do_bench(
        lambda: _matmul_kernel[grid](a, b, c, M, N, K,
                                     a.stride(0), a.stride(1),
                                     b.stride(0), b.stride(1),
                                     c.stride(0), c.stride(1),
                                     BLOCK_M=64, BLOCK_N=64, BLOCK_K=32))
    tflops = (2 * M * N * K) / (ms * 1e-3) / 1e12
    torch_ms = triton.testing.do_bench(lambda: a @ b)
    torch_tflops = (2 * M * N * K) / (torch_ms * 1e-3) / 1e12
    return max_err, tflops, torch_tflops


def probe():
    props = torch.cuda.get_device_properties(0)
    print(f"GPU:             {props.name}")
    print(f"Compute cap:     {props.major}.{props.minor}")
    print(f"SMs:             {props.multi_processor_count}")
    print(f"Total VRAM:      {props.total_memory / 1e9:.2f} GB")
    print(f"Shared mem/blk:  {props.shared_memory_per_block} B")
    print(f"Torch:           {torch.__version__}")
    print(f"Triton:          {triton.__version__}")
    print()

    for dtype in (torch.float32, torch.float16):
        try:
            max_err, tflops, torch_tflops = try_dot(dtype)
            print(f"tl.dot {str(dtype):<16} OK   max_err={max_err:.2e}  "
                  f"triton={tflops:.2f} TFLOPs  torch={torch_tflops:.2f} TFLOPs")
        except Exception as exc:
            print(f"tl.dot {str(dtype):<16} FAIL {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    probe()
```

- [ ] **Step 2: Run the probe on the 1650 Ti**

Run: `python scripts/probe_hardware.py`

Record the full output. Three outcomes and their consequences:
- **Both dtypes OK, triton within ~3× of torch:** all work stays local.
- **fp32 OK, fp16 FAIL:** expected on sm_75 without tensor cores. fp32 work local; fp16 experiments move to Modal.
- **Both FAIL:** all `tl.dot` tasks (10, 12, 13, 14, 15, 16, 17) move to Modal. Memory-bound tasks stay local.

- [ ] **Step 3: Record the result in the findings directory**

Create `docs/findings/00-hardware.md` containing the verbatim probe output and one paragraph stating which tasks run where. This decision is referenced by every later benchmark.

- [ ] **Step 4: Commit**

```bash
git add scripts/probe_hardware.py docs/findings/00-hardware.md
git commit -m "feat(bench): add tl.dot hardware capability probe"
```

---

### Task 2: Model config and variant registry

The registry is the mechanism every later task depends on. It must exist and be tested first.

**Files:**
- Create: `model/config.py`, `model/registry.py`, `tests/test_registry.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `ViTConfig` frozen dataclass with fields `image_size=32`, `patch_size=4`, `in_channels=3`, `dim=192`, `depth=6`, `heads=3`, `mlp_hidden=768`, `num_classes=10`, `eps=1e-5`; properties `num_patches -> int` (64), `head_dim -> int` (64), `scale -> float` (`head_dim ** -0.5`).
  - `Component` — `StrEnum` with members `LAYERNORM`, `GELU`, `SOFTMAX`, `LINEAR`, `ATTENTION`, `MLP`, `BLOCK`.
  - `register(component: Component, variant: str) -> Callable` decorator.
  - `get(component: Component, variant: str) -> Callable`.
  - `variants(component: Component) -> list[str]`.
  - `VariantConfig` frozen dataclass, one `str` field per `Component` member (all defaulting to `"torch"`), validating every field against the registry in `__post_init__`; method `resolve(component: Component) -> Callable`.

**Design note:** the spec called for "an enum per component." A validated `VariantConfig` achieves the identical guarantee — invalid configurations fail at construction — without duplicating the registry's contents into a parallel enum that would drift out of sync. The valid set is exactly the registry, by construction.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_registry.py
import pytest
from model.config import ViTConfig
from model.registry import Component, VariantConfig, get, register, variants


def test_config_derived_shapes():
    cfg = ViTConfig()
    assert cfg.num_patches == 64
    assert cfg.head_dim == 64
    assert cfg.dim == cfg.heads * cfg.head_dim
    assert cfg.scale == pytest.approx(0.125)


def test_register_and_get():
    @register(Component.GELU, "unit_test_variant")
    def _impl(x):
        return x
    assert get(Component.GELU, "unit_test_variant") is _impl
    assert "unit_test_variant" in variants(Component.GELU)


def test_duplicate_registration_rejected():
    @register(Component.GELU, "dupe_test")
    def _a(x):
        return x
    with pytest.raises(ValueError, match="already registered"):
        @register(Component.GELU, "dupe_test")
        def _b(x):
            return x


def test_unknown_variant_lists_alternatives():
    with pytest.raises(KeyError, match="available"):
        get(Component.GELU, "does_not_exist")


def test_variant_config_defaults_to_torch():
    cfg = VariantConfig()
    assert cfg.gelu == "torch"
    assert cfg.block == "torch"


def test_invalid_variant_fails_at_construction():
    with pytest.raises(ValueError, match="not a registered variant"):
        VariantConfig(gelu="nonexistent_kernel")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_registry.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'model.config'`

- [ ] **Step 3: Implement the config**

```python
# model/config.py
from dataclasses import dataclass


@dataclass(frozen=True)
class ViTConfig:
    image_size: int = 32
    patch_size: int = 4
    in_channels: int = 3
    dim: int = 192
    depth: int = 6
    heads: int = 3
    mlp_hidden: int = 768
    num_classes: int = 10
    eps: float = 1e-5

    @property
    def num_patches(self) -> int:
        side = self.image_size // self.patch_size
        return side * side

    @property
    def head_dim(self) -> int:
        return self.dim // self.heads

    @property
    def scale(self) -> float:
        return self.head_dim ** -0.5
```

- [ ] **Step 4: Implement the registry**

```python
# model/registry.py
from collections.abc import Callable
from dataclasses import dataclass, fields
from enum import StrEnum


class Component(StrEnum):
    LAYERNORM = "layernorm"
    GELU = "gelu"
    SOFTMAX = "softmax"
    LINEAR = "linear"
    ATTENTION = "attention"
    MLP = "mlp"
    BLOCK = "block"


_REGISTRY: dict[tuple[Component, str], Callable] = {}


def register(component: Component, variant: str) -> Callable:
    def decorator(fn: Callable) -> Callable:
        key = (component, variant)
        if key in _REGISTRY:
            raise ValueError(
                f"{component}/{variant} is already registered to "
                f"{_REGISTRY[key].__qualname__}")
        _REGISTRY[key] = fn
        return fn
    return decorator


def get(component: Component, variant: str) -> Callable:
    try:
        return _REGISTRY[(component, variant)]
    except KeyError:
        raise KeyError(
            f"no variant {variant!r} for {component}; "
            f"available: {variants(component)}") from None


def variants(component: Component) -> list[str]:
    return sorted(v for (c, v) in _REGISTRY if c == component)


@dataclass(frozen=True)
class VariantConfig:
    layernorm: str = "torch"
    gelu: str = "torch"
    softmax: str = "torch"
    linear: str = "torch"
    attention: str = "torch"
    mlp: str = "torch"
    block: str = "torch"

    def __post_init__(self) -> None:
        for field in fields(self):
            component = Component(field.name)
            variant = getattr(self, field.name)
            if (component, variant) not in _REGISTRY:
                raise ValueError(
                    f"{variant!r} is not a registered variant of {component}; "
                    f"available: {variants(component)}")

    def resolve(self, component: Component) -> Callable:
        return get(component, getattr(self, component.value))
```

- [ ] **Step 5: Run tests — expect the VariantConfig tests to still fail**

Run: `pytest tests/test_registry.py -v`
Expected: registry tests PASS; `test_variant_config_defaults_to_torch` FAILs because no `torch` variants are registered yet. This is correct — Task 3 registers them. Leave that test failing and note it; it turns green at the end of Task 3.

- [ ] **Step 6: Commit**

```bash
git add model/config.py model/registry.py tests/test_registry.py
git commit -m "feat(model): add ViT config and variant registry"
```

---

### Task 3: PyTorch reference implementation

The oracle. Every correctness assertion in the project compares against this, so it stays deliberately unclever.

**Files:**
- Create: `model/baseline/layers.py`, `model/baseline/__init__.py`, `tests/conftest.py`, `tests/test_baseline.py`

**Interfaces:**
- Consumes: `Component`, `register` from Task 2.
- Produces, all registered under variant `"torch"` and importable directly:
  - `layernorm(x: Tensor, weight: Tensor, bias: Tensor, eps: float) -> Tensor` — `x` is `[..., D]`, `weight`/`bias` are `[D]`.
  - `gelu(x: Tensor) -> Tensor` — tanh approximation.
  - `softmax(x: Tensor) -> Tensor` — over the last dimension only.
  - `linear(x: Tensor, weight: Tensor, bias: Tensor | None) -> Tensor` — `weight` is `[out, in]`.
  - `attention(q, k, v, scale: float) -> Tensor` — each `[B, H, S, Dh]`.
  - `mlp(x, w1, b1, w2, b2) -> Tensor` — `w1` is `[hidden, dim]`, `w2` is `[dim, hidden]`.
  - `block(x, ln1_w, ln1_b, qkv_w, qkv_b, proj_w, proj_b, ln2_w, ln2_b, w1, b1, w2, b2, heads: int, scale: float, eps: float) -> Tensor` — pre-norm.
  - `tests/conftest.py` exposes fixtures `device`, `cfg`, and `TOLERANCES: dict[str, dict[str, float]]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/conftest.py
import pytest
import torch
from model.config import ViTConfig

# Declared per kernel and justified. Reduction order differs from torch's,
# so bitwise equality is impossible. NEVER loosen these to make a test pass:
# a test needing more slack is a bug report, not a tuning knob.
TOLERANCES = {
    "layernorm": {"rtol": 1e-4, "atol": 1e-5},
    "gelu":      {"rtol": 1e-5, "atol": 1e-6},
    "softmax":   {"rtol": 1e-5, "atol": 1e-6},
    "linear":    {"rtol": 1e-4, "atol": 1e-4},
    "attention": {"rtol": 1e-4, "atol": 1e-4},
    "mlp":       {"rtol": 1e-4, "atol": 1e-4},
    "block":     {"rtol": 1e-3, "atol": 1e-4},
}


@pytest.fixture(scope="session")
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    return torch.device("cuda")


@pytest.fixture(scope="session")
def cfg():
    return ViTConfig()
```

```python
# tests/test_baseline.py
import torch
import torch.nn.functional as F
from model.baseline.layers import attention, gelu, layernorm, linear, mlp, softmax
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_baseline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'model.baseline.layers'`

- [ ] **Step 3: Implement the reference layers**

```python
# model/baseline/layers.py
"""Pure-PyTorch reference. This is the oracle for every correctness test
and the baseline for every benchmark, so it stays deliberately boring:
plain functional calls, no optimization, no cleverness.
"""
import torch
import torch.nn.functional as F
from torch import Tensor

from model.registry import Component, register


@register(Component.LAYERNORM, "torch")
def layernorm(x: Tensor, weight: Tensor, bias: Tensor, eps: float) -> Tensor:
    return F.layer_norm(x, (x.shape[-1],), weight, bias, eps)


@register(Component.GELU, "torch")
def gelu(x: Tensor) -> Tensor:
    return F.gelu(x, approximate="tanh")


@register(Component.SOFTMAX, "torch")
def softmax(x: Tensor) -> Tensor:
    return F.softmax(x, dim=-1)


@register(Component.LINEAR, "torch")
def linear(x: Tensor, weight: Tensor, bias: Tensor | None) -> Tensor:
    return F.linear(x, weight, bias)


@register(Component.ATTENTION, "torch")
def attention(q: Tensor, k: Tensor, v: Tensor, scale: float) -> Tensor:
    scores = softmax((q @ k.transpose(-2, -1)) * scale)
    return scores @ v


@register(Component.MLP, "torch")
def mlp(x: Tensor, w1: Tensor, b1: Tensor, w2: Tensor, b2: Tensor) -> Tensor:
    return linear(gelu(linear(x, w1, b1)), w2, b2)


@register(Component.BLOCK, "torch")
def block(x: Tensor, ln1_w: Tensor, ln1_b: Tensor,
          qkv_w: Tensor, qkv_b: Tensor, proj_w: Tensor, proj_b: Tensor,
          ln2_w: Tensor, ln2_b: Tensor,
          w1: Tensor, b1: Tensor, w2: Tensor, b2: Tensor,
          heads: int, scale: float, eps: float) -> Tensor:
    batch, seq, dim = x.shape
    head_dim = dim // heads

    normed = layernorm(x, ln1_w, ln1_b, eps)
    qkv = linear(normed, qkv_w, qkv_b)
    qkv = qkv.reshape(batch, seq, 3, heads, head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    attended = attention(q, k, v, scale)
    attended = attended.transpose(1, 2).reshape(batch, seq, dim)
    x = x + linear(attended, proj_w, proj_b)

    return x + mlp(layernorm(x, ln2_w, ln2_b, eps), w1, b1, w2, b2)
```

```python
# model/baseline/__init__.py
from model.baseline import layers  # noqa: F401  (registers "torch" variants)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_baseline.py tests/test_registry.py -v`
Expected: all PASS, including the `VariantConfig` tests deferred from Task 2. If those still fail, `model.baseline` was not imported — add the import to `tests/conftest.py`.

- [ ] **Step 5: Commit**

```bash
git add model/baseline tests/conftest.py tests/test_baseline.py
git commit -m "feat(model): add PyTorch reference layers as test oracle"
```

---

### Task 4: Reference ViT and frozen checkpoint

Produces the trained weights every later accuracy check compares against. Freeze this early: retraining midway makes end-to-end numbers incomparable across time.

**Files:**
- Create: `model/baseline/vit.py`, `scripts/train_reference.py`, `tests/test_vit_baseline.py`

**Interfaces:**
- Consumes: `ViTConfig`, baseline layers from Task 3.
- Produces:
  - `VisionTransformer(cfg: ViTConfig)` — `nn.Module` with `forward(images: Tensor) -> Tensor`, images `[B, 3, 32, 32]`, logits `[B, 10]`.
  - Named parameters per block, accessed later by the Triton path: `blocks.{i}.ln1_w`, `ln1_b`, `qkv_w`, `qkv_b`, `proj_w`, `proj_b`, `ln2_w`, `ln2_b`, `w1`, `b1`, `w2`, `b2`.
  - `data/checkpoint.pt` — dict with keys `state_dict`, `cfg`, `test_accuracy`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_vit_baseline.py
import torch
from model.baseline.vit import VisionTransformer
from model.config import ViTConfig


def test_forward_shape(device):
    cfg = ViTConfig()
    model = VisionTransformer(cfg).to(device).eval()
    images = torch.randn(4, 3, 32, 32, device=device)
    with torch.no_grad():
        logits = model(images)
    assert logits.shape == (4, 10)
    assert torch.isfinite(logits).all()


def test_patch_embedding_produces_64_tokens(device):
    cfg = ViTConfig()
    model = VisionTransformer(cfg).to(device).eval()
    images = torch.randn(2, 3, 32, 32, device=device)
    with torch.no_grad():
        tokens = model.embed(images)
    assert tokens.shape == (2, 64, 192)


def test_no_cls_token_parameter():
    model = VisionTransformer(ViTConfig())
    assert not any("cls" in name for name, _ in model.named_parameters())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_vit_baseline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'model.baseline.vit'`

- [ ] **Step 3: Implement the reference ViT**

```python
# model/baseline/vit.py
import torch
from torch import Tensor, nn

from model.baseline.layers import block, layernorm
from model.config import ViTConfig


class BlockParams(nn.Module):
    """Raw parameter tensors. Held as plain Parameters rather than nn.Linear
    modules so the Triton kernels can consume them without unwrapping."""

    def __init__(self, cfg: ViTConfig):
        super().__init__()
        dim, hidden = cfg.dim, cfg.mlp_hidden
        self.ln1_w = nn.Parameter(torch.ones(dim))
        self.ln1_b = nn.Parameter(torch.zeros(dim))
        self.qkv_w = nn.Parameter(torch.empty(3 * dim, dim))
        self.qkv_b = nn.Parameter(torch.zeros(3 * dim))
        self.proj_w = nn.Parameter(torch.empty(dim, dim))
        self.proj_b = nn.Parameter(torch.zeros(dim))
        self.ln2_w = nn.Parameter(torch.ones(dim))
        self.ln2_b = nn.Parameter(torch.zeros(dim))
        self.w1 = nn.Parameter(torch.empty(hidden, dim))
        self.b1 = nn.Parameter(torch.zeros(hidden))
        self.w2 = nn.Parameter(torch.empty(dim, hidden))
        self.b2 = nn.Parameter(torch.zeros(hidden and dim))
        for weight in (self.qkv_w, self.proj_w, self.w1, self.w2):
            nn.init.trunc_normal_(weight, std=0.02)


class VisionTransformer(nn.Module):
    def __init__(self, cfg: ViTConfig):
        super().__init__()
        self.cfg = cfg
        patch_elems = cfg.in_channels * cfg.patch_size ** 2
        self.patch_w = nn.Parameter(torch.empty(cfg.dim, patch_elems))
        self.patch_b = nn.Parameter(torch.zeros(cfg.dim))
        self.pos = nn.Parameter(torch.zeros(1, cfg.num_patches, cfg.dim))
        self.blocks = nn.ModuleList(BlockParams(cfg) for _ in range(cfg.depth))
        self.norm_w = nn.Parameter(torch.ones(cfg.dim))
        self.norm_b = nn.Parameter(torch.zeros(cfg.dim))
        self.head_w = nn.Parameter(torch.empty(cfg.num_classes, cfg.dim))
        self.head_b = nn.Parameter(torch.zeros(cfg.num_classes))
        nn.init.trunc_normal_(self.patch_w, std=0.02)
        nn.init.trunc_normal_(self.head_w, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)

    def embed(self, images: Tensor) -> Tensor:
        cfg = self.cfg
        patches = images.unfold(2, cfg.patch_size, cfg.patch_size) \
                        .unfold(3, cfg.patch_size, cfg.patch_size)
        patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(
            images.shape[0], cfg.num_patches, -1)
        return torch.nn.functional.linear(
            patches, self.patch_w, self.patch_b) + self.pos

    def forward(self, images: Tensor) -> Tensor:
        cfg = self.cfg
        x = self.embed(images)
        for blk in self.blocks:
            x = block(x, blk.ln1_w, blk.ln1_b, blk.qkv_w, blk.qkv_b,
                      blk.proj_w, blk.proj_b, blk.ln2_w, blk.ln2_b,
                      blk.w1, blk.b1, blk.w2, blk.b2,
                      cfg.heads, cfg.scale, cfg.eps)
        x = layernorm(x, self.norm_w, self.norm_b, cfg.eps)
        return torch.nn.functional.linear(x.mean(dim=1), self.head_w,
                                          self.head_b)
```

Note: `torch.empty(hidden and dim)` above is a deliberate trap to catch — it should read `torch.zeros(dim)`. Fix it while implementing; the shape test will catch it if you do not.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_vit_baseline.py -v`
Expected: PASS

- [ ] **Step 5: Write the training script**

```python
# scripts/train_reference.py
"""Trains the reference ViT and freezes the checkpoint.

Run this ONCE. Every accuracy comparison for the life of the project
comes from the resulting file; retraining invalidates historical numbers.
"""
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model.baseline.vit import VisionTransformer
from model.config import ViTConfig

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def loaders(root, batch_size):
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    train = datasets.CIFAR10(root, train=True, download=True, transform=train_tf)
    test = datasets.CIFAR10(root, train=False, download=True, transform=test_tf)
    return (DataLoader(train, batch_size, shuffle=True, num_workers=4,
                       drop_last=True),
            DataLoader(test, 512, shuffle=False, num_workers=4))


@torch.no_grad()
def accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    for images, labels in loader:
        preds = model(images.to(device)).argmax(-1).cpu()
        correct += (preds == labels).sum().item()
        total += labels.numel()
    return correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--out", default="data/checkpoint.pt")
    args = parser.parse_args()

    device = torch.device("cuda")
    cfg = ViTConfig()
    model = VisionTransformer(cfg).to(device)
    train_loader, test_loader = loaders(args.data_root, args.batch_size)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, epochs=args.epochs,
        steps_per_epoch=len(train_loader))

    for epoch in range(args.epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            loss = F.cross_entropy(model(images), labels, label_smoothing=0.1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        if (epoch + 1) % 10 == 0:
            print(f"epoch {epoch + 1}: test acc {accuracy(model, test_loader, device):.4f}")

    final = accuracy(model, test_loader, device)
    torch.save({"state_dict": model.state_dict(), "cfg": cfg,
                "test_accuracy": final}, args.out)
    print(f"saved {args.out} with test accuracy {final:.4f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Train the checkpoint**

Run: `python scripts/train_reference.py --epochs 100`
Expected: final test accuracy in the 0.83–0.90 range. Below 0.80 means something is wrong with augmentation or the schedule — investigate before proceeding, since a weak checkpoint makes the end-to-end agreement gate less meaningful.

If the 1650 Ti is too slow, run this on Modal and copy `data/checkpoint.pt` back.

- [ ] **Step 7: Commit the checkpoint**

```bash
git add model/baseline/vit.py scripts/train_reference.py tests/test_vit_baseline.py data/checkpoint.pt
git commit -m "feat(model): add reference ViT and frozen CIFAR-10 checkpoint"
```

---

### Task 5: Benchmark harness

Produces latency numbers that survive thermal drift on a laptop GPU. Every later task's measurements flow through this.

**Files:**
- Create: `bench/clocks.py`, `bench/harness.py`, `scripts/lock_clocks.sh`, `tests/test_harness.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `bench.clocks.telemetry() -> tuple[int, int]` returning `(sm_clock_mhz, temp_c)`; returns `(-1, -1)` if `nvidia-smi` is unavailable.
  - `bench.clocks.locked_clock_mhz() -> int | None`.
  - `bench.harness.compare(arms: dict[str, Callable[[], object]], reps: int = 30) -> dict[str, list[float]]` — runs arms **interleaved at the rep level**, returns per-arm millisecond samples.
  - `bench.harness.Measurement` frozen dataclass with fields matching the CSV schema.
  - `bench.harness.record(rows: list[Measurement], path: str) -> None` — appends, writing a header if the file is new.
  - CSV columns, in order: `timestamp, commit_sha, gpu, kernel, variant, batch, dtype, latency_ms_median, latency_ms_p10, latency_ms_p90, bytes_theoretical, achieved_gbps, sm_clock_mhz, temp_c, flagged`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_harness.py
import csv
import time
from bench.harness import Measurement, compare, record


def test_compare_runs_arms_interleaved():
    order = []

    def make(name):
        def arm():
            order.append(name)
        return arm

    compare({"a": make("a"), "b": make("b")}, reps=4)
    # Interleaving means neighbours differ; batching would give aaaabbbb.
    pairs = list(zip(order, order[1:]))
    assert sum(1 for x, y in pairs if x == y) < len(pairs) / 2


def test_compare_returns_samples_per_arm():
    samples = compare({"a": lambda: None, "b": lambda: None}, reps=5)
    assert set(samples) == {"a", "b"}
    assert all(len(v) == 5 for v in samples.values())


def test_record_writes_header_once(tmp_path):
    path = tmp_path / "out.csv"
    row = Measurement(
        timestamp="2026-08-13T00:00:00", commit_sha="abc1234", gpu="test",
        kernel="layernorm", variant="triton", batch=8, dtype="float32",
        latency_ms_median=1.0, latency_ms_p10=0.9, latency_ms_p90=1.1,
        bytes_theoretical=1024, achieved_gbps=1.0,
        sm_clock_mhz=1500, temp_c=60, flagged=False)
    record([row], str(path))
    record([row], str(path))
    with open(path) as handle:
        rows = list(csv.reader(handle))
    assert rows[0][0] == "timestamp"
    assert len(rows) == 3


def test_flagged_when_clock_deviates():
    row = Measurement.build(
        kernel="k", variant="v", batch=1, dtype="float32",
        samples=[1.0, 1.0, 1.0], bytes_theoretical=1024,
        gpu="test", sm_clock_mhz=1000, temp_c=80, locked_clock_mhz=1500)
    assert row.flagged is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_harness.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'bench.harness'`

- [ ] **Step 3: Implement clock telemetry**

```python
# bench/clocks.py
import subprocess


def _query(field: str) -> int | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    return int(out.stdout.strip().splitlines()[0])


def telemetry() -> tuple[int, int]:
    return _query("clocks.sm") or -1, _query("temperature.gpu") or -1


def locked_clock_mhz() -> int | None:
    """Applications clock, set by `nvidia-smi -lgc`. None if unlocked."""
    return _query("clocks.applications.graphics")
```

```bash
# scripts/lock_clocks.sh
#!/usr/bin/env bash
# Lock clocks before benchmarking. Requires root.
# Pick a graphics clock the card can sustain thermally -- for a laptop
# 1650 Ti that is well below boost. Check supported values with:
#   nvidia-smi --query-supported-clocks=gr --format=csv
set -euo pipefail
CLOCK="${1:-1200}"
sudo nvidia-smi -pm 1
sudo nvidia-smi -lgc "${CLOCK},${CLOCK}"
echo "locked graphics clock to ${CLOCK} MHz; reset with: sudo nvidia-smi -rgc"
```

- [ ] **Step 4: Implement the harness**

```python
# bench/harness.py
"""Latency measurement. Counters come from bench/profile.py and are written
to a different file -- ncu serializes and replays kernels, so its durations
are not comparable to anything here.
"""
import csv
import os
import statistics
import subprocess
from collections.abc import Callable
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone

import torch
import triton

from bench.clocks import locked_clock_mhz, telemetry

CLOCK_DEVIATION_TOLERANCE = 0.05


def commit_sha() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def gpu_name() -> str:
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"


def compare(arms: dict[str, Callable[[], object]],
            reps: int = 30) -> dict[str, list[float]]:
    """Time each arm, interleaved at the rep level.

    Batching all reps of one arm before the next measures the heatsink:
    the second arm runs hotter. Interleaving spreads thermal drift across
    both arms instead of loading it entirely onto whichever ran second.
    """
    for arm in arms.values():
        for _ in range(5):
            arm()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    samples: dict[str, list[float]] = {name: [] for name in arms}
    for _ in range(reps):
        for name, arm in arms.items():
            samples[name].append(triton.testing.do_bench(arm, warmup=0, rep=1))
    return samples


@dataclass(frozen=True)
class Measurement:
    timestamp: str
    commit_sha: str
    gpu: str
    kernel: str
    variant: str
    batch: int
    dtype: str
    latency_ms_median: float
    latency_ms_p10: float
    latency_ms_p90: float
    bytes_theoretical: int
    achieved_gbps: float
    sm_clock_mhz: int
    temp_c: int
    flagged: bool

    @classmethod
    def build(cls, *, kernel: str, variant: str, batch: int, dtype: str,
              samples: list[float], bytes_theoretical: int,
              gpu: str | None = None, sm_clock_mhz: int | None = None,
              temp_c: int | None = None,
              locked_clock_mhz: int | None = None) -> "Measurement":
        ordered = sorted(samples)
        median = statistics.median(ordered)
        if sm_clock_mhz is None or temp_c is None:
            sm_clock_mhz, temp_c = telemetry()
        flagged = False
        if locked_clock_mhz:
            drift = abs(sm_clock_mhz - locked_clock_mhz) / locked_clock_mhz
            flagged = drift > CLOCK_DEVIATION_TOLERANCE
        return cls(
            timestamp=datetime.now(timezone.utc).isoformat(),
            commit_sha=commit_sha(),
            gpu=gpu if gpu is not None else gpu_name(),
            kernel=kernel, variant=variant, batch=batch, dtype=dtype,
            latency_ms_median=median,
            latency_ms_p10=ordered[int(0.1 * (len(ordered) - 1))],
            latency_ms_p90=ordered[int(0.9 * (len(ordered) - 1))],
            bytes_theoretical=bytes_theoretical,
            achieved_gbps=bytes_theoretical / (median * 1e-3) / 1e9,
            sm_clock_mhz=sm_clock_mhz, temp_c=temp_c, flagged=flagged)


def record(rows: list[Measurement], path: str) -> None:
    columns = [f.name for f in fields(Measurement)]
    is_new = not os.path.exists(path) or os.path.getsize(path) == 0
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        if is_new:
            writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_harness.py -v`
Expected: PASS. `test_compare_runs_arms_interleaved` runs on CPU without CUDA; if `do_bench` requires CUDA in your Triton version, mark it `@pytest.mark.skipif(not torch.cuda.is_available())` and verify interleaving through the `order` list alone.

- [ ] **Step 6: Commit**

```bash
git add bench/clocks.py bench/harness.py scripts/lock_clocks.sh tests/test_harness.py
git commit -m "feat(bench): add interleaved benchmark harness with clock telemetry"
```

---

### Task 6: Nsight profiling wrapper

**Files:**
- Create: `bench/profile.py`, `tests/test_profile.py`

**Interfaces:**
- Consumes: `commit_sha`, `gpu_name` from `bench.harness`.
- Produces:
  - `bench.profile.METRICS: list[str]` — the seven-metric set.
  - `bench.profile.parse_ncu_csv(text: str) -> list[dict[str, str]]` — parses `ncu --csv` output, skipping preamble lines before the header row.
  - `bench.profile.profile_kernel(module: str, kernel: str, variant: str, batch: int, dtype: str, launch_skip: int = 5, launch_count: int = 1) -> list[dict]` — invokes `ncu` on `python -m <module>` and returns parsed rows annotated with the run's identity.
  - `bench.profile.record_counters(rows: list[dict], path: str) -> None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_profile.py
from bench.profile import METRICS, parse_ncu_csv

SAMPLE = '''==PROF== Connected to process 1234
==PROF== Profiling "layernorm_kernel" - 0: 0%....50%....100%
"ID","Kernel Name","Metric Name","Metric Unit","Metric Value"
"0","layernorm_kernel","dram__bytes_read.sum","byte","98304"
"0","layernorm_kernel","launch__registers_per_thread","register/thread","32"
==PROF== Disconnected from process 1234
'''


def test_parse_skips_preamble():
    rows = parse_ncu_csv(SAMPLE)
    assert len(rows) == 2
    assert rows[0]["Metric Name"] == "dram__bytes_read.sum"
    assert rows[0]["Metric Value"] == "98304"


def test_parse_returns_empty_for_no_header():
    assert parse_ncu_csv("==PROF== nothing here\n") == []


def test_metric_set_includes_spill_counters():
    assert "l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum" in METRICS
    assert "l1tex__t_bytes_pipe_lsu_mem_local_op_st.sum" in METRICS
    assert "sm__warps_active.avg.pct_of_peak_sustained_active" in METRICS
    assert "launch__registers_per_thread" in METRICS
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_profile.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'bench.profile'`

- [ ] **Step 3: Implement the profiler wrapper**

```python
# bench/profile.py
"""Hardware counter collection via Nsight Compute.

ncu serializes execution and replays each kernel to gather counter sets,
so its reported durations are inflated and meaningless as performance
numbers. Latency lives in bench/harness.py and a separate CSV; nothing
here produces a timing.

Setup, once, on the profiling host:
    echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' \\
        | sudo tee /etc/modprobe.d/nvidia-profiling.conf
    sudo update-initramfs -u && sudo reboot
"""
import csv
import io
import os
import subprocess
from datetime import datetime, timezone

from bench.harness import commit_sha, gpu_name

METRICS = [
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "l1tex__t_sector_hit_rate.pct",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "launch__registers_per_thread",
    "l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum",
    "l1tex__t_bytes_pipe_lsu_mem_local_op_st.sum",
]

COUNTER_COLUMNS = [
    "timestamp", "commit_sha", "gpu", "kernel", "variant", "batch", "dtype",
    "kernel_name", "metric", "unit", "value",
]


def parse_ncu_csv(text: str) -> list[dict[str, str]]:
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.startswith('"ID"'):
            reader = csv.DictReader(io.StringIO("\n".join(lines[index:])))
            return [row for row in reader if row.get("Metric Name")]
    return []


def profile_kernel(module: str, kernel: str, variant: str, batch: int,
                   dtype: str, launch_skip: int = 5,
                   launch_count: int = 1) -> list[dict]:
    """Profile a single steady-state launch.

    launch_skip avoids the cold first launch, whose counters reflect
    autotuning and cache-cold behaviour rather than steady state.
    """
    command = [
        "ncu", "--csv", "--target-processes", "all",
        "--launch-skip", str(launch_skip),
        "--launch-count", str(launch_count),
        "--metrics", ",".join(METRICS),
        "python", "-m", module,
        "--kernel", kernel, "--variant", variant,
        "--batch", str(batch), "--dtype", dtype,
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ncu failed ({result.returncode}).\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")

    stamp = datetime.now(timezone.utc).isoformat()
    sha, gpu = commit_sha(), gpu_name()
    return [{
        "timestamp": stamp, "commit_sha": sha, "gpu": gpu,
        "kernel": kernel, "variant": variant, "batch": batch, "dtype": dtype,
        "kernel_name": row["Kernel Name"], "metric": row["Metric Name"],
        "unit": row.get("Metric Unit", ""), "value": row["Metric Value"],
    } for row in parse_ncu_csv(result.stdout)]


def record_counters(rows: list[dict], path: str) -> None:
    is_new = not os.path.exists(path) or os.path.getsize(path) == 0
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COUNTER_COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerows(rows)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_profile.py -v`
Expected: PASS

- [ ] **Step 5: Verify ncu permissions on the real host**

Run: `ncu --metrics dram__bytes_read.sum python -c "import torch; torch.randn(1024, device='cuda').sum()"`
Expected: a counter table. If you see `ERR_NVGPUCTRPERM`, apply the modprobe fix in the module docstring and reboot.

- [ ] **Step 6: Commit**

```bash
git add bench/profile.py tests/test_profile.py
git commit -m "feat(bench): add ncu counter collection wrapper"
```

---

### Task 7: Triton LayerNorm

First real kernel. Rung 1 of the ladder — establishes the bandwidth-bound floor.

**Files:**
- Create: `model/kernels/__init__.py`, `model/kernels/layernorm.py`, `tests/test_layernorm.py`

**Interfaces:**
- Consumes: `Component`, `register`; `TOLERANCES` from `tests/conftest.py`.
- Produces: `layernorm(x, weight, bias, eps) -> Tensor`, registered as `(LAYERNORM, "triton")`. Same signature as the baseline. Accepts non-contiguous `x`; rows are the flattened leading dimensions and the last dimension is normalized.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_layernorm.py
import pytest
import torch
import torch.nn.functional as F
from model.kernels.layernorm import layernorm
from tests.conftest import TOLERANCES

TOL = TOLERANCES["layernorm"]


@pytest.mark.parametrize("shape", [
    (1, 64, 192),      # batch 1 -- grid edge
    (8, 64, 192),      # typical
    (512, 64, 192),    # large batch
    (4, 64, 192),
])
def test_matches_torch(device, shape):
    x = torch.randn(shape, device=device)
    w = torch.randn(shape[-1], device=device)
    b = torch.randn(shape[-1], device=device)
    expected = F.layer_norm(x, (shape[-1],), w, b, 1e-5)
    torch.testing.assert_close(layernorm(x, w, b, 1e-5), expected, **TOL)


def test_non_power_of_two_feature_dim(device):
    """D=192 is not a power of two -- the masked partial block is the most
    likely source of silent garbage in every kernel in this project."""
    x = torch.randn(4, 64, 192, device=device)
    w = torch.ones(192, device=device)
    b = torch.zeros(192, device=device)
    out = layernorm(x, w, b, 1e-5)
    torch.testing.assert_close(out, F.layer_norm(x, (192,), w, b, 1e-5), **TOL)
    torch.testing.assert_close(out.mean(-1),
                               torch.zeros(4, 64, device=device), atol=1e-5,
                               rtol=1e-4)


def test_non_contiguous_input(device):
    """Attention transposes before normalizing in some variants, so kernels
    receive strided tensors -- not freshly allocated contiguous ones."""
    base = torch.randn(4, 192, 64, device=device)
    x = base.transpose(1, 2)
    assert not x.is_contiguous()
    w = torch.randn(192, device=device)
    b = torch.randn(192, device=device)
    torch.testing.assert_close(layernorm(x, w, b, 1e-5),
                               F.layer_norm(x, (192,), w, b, 1e-5), **TOL)


def test_near_zero_variance_uses_eps(device):
    x = torch.full((4, 64, 192), 3.0, device=device)
    w = torch.ones(192, device=device)
    b = torch.zeros(192, device=device)
    out = layernorm(x, w, b, 1e-5)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, torch.zeros_like(out), atol=1e-4, rtol=0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_layernorm.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'model.kernels.layernorm'`

- [ ] **Step 3: Implement the kernel**

```python
# model/kernels/layernorm.py
import torch
import triton
import triton.language as tl
from torch import Tensor

from model.registry import Component, register


@triton.jit
def _layernorm_kernel(x_ptr, w_ptr, b_ptr, out_ptr,
                      stride_row, n_cols, eps,
                      BLOCK: tl.constexpr):
    row = tl.program_id(0)
    x_row = x_ptr + row * stride_row
    out_row = out_ptr + row * stride_row

    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    # `other=0.0` matters twice: masked lanes must not pollute the sum, and
    # they must not fault. D=192 with BLOCK=256 leaves 64 masked lanes.
    x = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)

    mean = tl.sum(x, axis=0) / n_cols
    centered = tl.where(mask, x - mean, 0.0)
    var = tl.sum(centered * centered, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(var + eps)

    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_row + cols, centered * rstd * w + b, mask=mask)


@register(Component.LAYERNORM, "triton")
def layernorm(x: Tensor, weight: Tensor, bias: Tensor, eps: float) -> Tensor:
    x_flat = x.reshape(-1, x.shape[-1]).contiguous()
    rows, n_cols = x_flat.shape
    out = torch.empty_like(x_flat)
    block = triton.next_power_of_2(n_cols)
    _layernorm_kernel[(rows,)](
        x_flat, weight, bias, out,
        x_flat.stride(0), n_cols, eps,
        BLOCK=block,
        num_warps=4 if block <= 512 else 8)
    return out.reshape(x.shape)
```

```python
# model/kernels/__init__.py
from model.kernels import layernorm  # noqa: F401  (registers triton variants)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_layernorm.py -v`
Expected: PASS. If `test_non_contiguous_input` fails, the `.contiguous()` call is missing or the stride is wrong.

- [ ] **Step 5: Benchmark against the baseline**

Create `bench/run_layernorm.py` following the `compare` interface from Task 5: build `x`, `w`, `b` for each batch in `{1, 8, 32, 128, 512}`, define arms `{"torch": ..., "triton": ...}`, call `compare`, build `Measurement`s with `bytes_theoretical = 2 * batch * 64 * 192 * 4` (one read, one write), and `record` to `bench/results/latency.csv`.

Run: `python -m bench.run_layernorm`
Expected: triton within ~1.2× of torch at large batch. Both are bandwidth-bound, so a large gap means a bug — check `achieved_gbps` against the card's ~192 GB/s.

- [ ] **Step 6: Commit**

```bash
git add model/kernels tests/test_layernorm.py bench/run_layernorm.py bench/results/latency.csv
git commit -m "feat(kernels): add Triton LayerNorm"
```

---

### Task 8: Triton GeLU

Rung 3. **Expected to show no improvement** — that is the point. A project reporting only wins has not measured anything.

**Files:**
- Create: `model/kernels/gelu.py`, `tests/test_gelu.py`, `docs/findings/01-elementwise.md`
- Modify: `model/kernels/__init__.py`

**Interfaces:**
- Consumes: `Component`, `register`.
- Produces: `gelu(x) -> Tensor` registered as `(GELU, "triton")`, tanh approximation matching `F.gelu(approximate="tanh")` exactly in formula.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_gelu.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_gelu.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement the kernel**

```python
# model/kernels/gelu.py
import torch
import triton
import triton.language as tl
from torch import Tensor

from model.registry import Component, register

SQRT_2_OVER_PI = 0.7978845608028654


@triton.jit
def _gelu_kernel(x_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    inner = 0.7978845608028654 * (x + 0.044715 * x * x * x)
    tl.store(out_ptr + offsets, 0.5 * x * (1.0 + tl.math.tanh(inner)),
             mask=mask)


@register(Component.GELU, "triton")
def gelu(x: Tensor) -> Tensor:
    x_flat = x.contiguous().reshape(-1)
    out = torch.empty_like(x_flat)
    n_elements = x_flat.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK"]),)
    _gelu_kernel[grid](x_flat, out, n_elements, BLOCK=1024)
    return out.reshape(x.shape)
```

Add `from model.kernels import gelu  # noqa: F401` to `model/kernels/__init__.py`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_gelu.py -v`
Expected: PASS

- [ ] **Step 5: Benchmark and profile**

Create `bench/run_gelu.py` mirroring `bench/run_layernorm.py`, with `bytes_theoretical = 2 * batch * 64 * 768 * 4`.

Run: `python -m bench.run_gelu`
Then: `python -c "from bench.profile import profile_kernel, record_counters; record_counters(profile_kernel('bench.run_gelu', 'gelu', 'triton', 128, 'float32'), 'bench/results/counters.csv')"`

Expected: triton and torch within noise of each other; both near peak bandwidth.

- [ ] **Step 6: Write the findings document**

Create `docs/findings/01-elementwise.md` recording measured latencies for both arms across the batch sweep, the achieved GB/s against the ~192 GB/s peak, and a paragraph answering: *if both implementations are bandwidth-bound and near peak, what could a Triton rewrite possibly have improved?* This is the reasoning that motivates fusion — it should be written down before the first fusion is attempted.

- [ ] **Step 7: Commit**

```bash
git add model/kernels/gelu.py model/kernels/__init__.py tests/test_gelu.py bench/run_gelu.py docs/findings/01-elementwise.md bench/results/
git commit -m "feat(kernels): add Triton GeLU and elementwise findings"
```

---

### Task 9: Triton Softmax

Rung 4. Rows are 64 floats = 256 B. **Expect torch to win at low batch** — launch overhead dominates.

**Files:**
- Create: `model/kernels/softmax.py`, `tests/test_softmax.py`
- Modify: `model/kernels/__init__.py`

**Interfaces:**
- Consumes: `Component`, `register`.
- Produces: `softmax(x) -> Tensor` registered as `(SOFTMAX, "triton")`, over the last dimension only. Input typically `[B, H, S, S]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_softmax.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_softmax.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement the kernel**

```python
# model/kernels/softmax.py
import torch
import triton
import triton.language as tl
from torch import Tensor

from model.registry import Component, register


@triton.jit
def _softmax_kernel(x_ptr, out_ptr, stride_row, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    # -inf on masked lanes so they lose the max and contribute exp(-inf)=0.
    x = tl.load(x_ptr + row * stride_row + cols, mask=mask,
                other=float("-inf")).to(tl.float32)
    x = x - tl.max(x, axis=0)
    numerator = tl.exp(x)
    numerator = tl.where(mask, numerator, 0.0)
    tl.store(out_ptr + row * stride_row + cols,
             numerator / tl.sum(numerator, axis=0), mask=mask)


@register(Component.SOFTMAX, "triton")
def softmax(x: Tensor) -> Tensor:
    x_flat = x.contiguous().reshape(-1, x.shape[-1])
    rows, n_cols = x_flat.shape
    out = torch.empty_like(x_flat)
    block = triton.next_power_of_2(n_cols)
    _softmax_kernel[(rows,)](x_flat, out, x_flat.stride(0), n_cols,
                             BLOCK=block, num_warps=4)
    return out.reshape(x.shape)
```

Add `from model.kernels import softmax  # noqa: F401` to `model/kernels/__init__.py`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_softmax.py -v`
Expected: PASS. A NaN failure in `test_large_magnitude_inputs_do_not_overflow` means the `other=float("-inf")` masking or the max-subtraction is wrong.

- [ ] **Step 5: Benchmark across the batch sweep**

Create `bench/run_softmax.py` with `bytes_theoretical = 2 * batch * 3 * 64 * 64 * 4`.

Run: `python -m bench.run_softmax`
Expected: torch faster at batch 1–8, triton competitive by batch 128. Record the crossover batch — it is the launch-overhead boundary and will be referenced when fusion results are interpreted.

- [ ] **Step 6: Commit**

```bash
git add model/kernels/softmax.py model/kernels/__init__.py tests/test_softmax.py bench/run_softmax.py bench/results/
git commit -m "feat(kernels): add Triton softmax with online max subtraction"
```

---

### Task 10: Triton Linear (tiled matmul)

Rung 5. Gated on Task 1: if `tl.dot` failed on the 1650 Ti, this and all subsequent matmul tasks run on Modal.

**Files:**
- Create: `model/kernels/linear.py`, `tests/test_linear.py`
- Modify: `model/kernels/__init__.py`

**Interfaces:**
- Consumes: `Component`, `register`.
- Produces: `linear(x, weight, bias) -> Tensor` registered as `(LINEAR, "triton")`. `x` is `[..., K]`, `weight` is `[N, K]`, `bias` is `[N]` or `None`. Output `[..., N]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_linear.py
import pytest
import torch
import torch.nn.functional as F
from model.kernels.linear import linear
from tests.conftest import TOLERANCES

TOL = TOLERANCES["linear"]

# (in_features, out_features) actually used by the model.
SHAPES = [(192, 576), (192, 192), (192, 768), (768, 192)]


@pytest.mark.parametrize("batch", [1, 8, 128])
@pytest.mark.parametrize("k,n", SHAPES)
def test_matches_torch(device, batch, k, n):
    x = torch.randn(batch, 64, k, device=device)
    w = torch.randn(n, k, device=device) * 0.05
    b = torch.randn(n, device=device)
    torch.testing.assert_close(linear(x, w, b), F.linear(x, w, b), **TOL)


def test_without_bias(device):
    x = torch.randn(4, 64, 192, device=device)
    w = torch.randn(192, 192, device=device) * 0.05
    torch.testing.assert_close(linear(x, w, None), F.linear(x, w, None), **TOL)


def test_non_contiguous_input(device):
    x = torch.randn(4, 192, 64, device=device).transpose(1, 2)
    assert not x.is_contiguous()
    w = torch.randn(192, 192, device=device) * 0.05
    b = torch.randn(192, device=device)
    torch.testing.assert_close(linear(x, w, b), F.linear(x, w, b), **TOL)


def test_non_power_of_two_k_dimension(device):
    """K=192 leaves a partial tile on the reduction axis; an unmasked load
    there silently adds garbage to the accumulator."""
    x = torch.randn(2, 64, 192, device=device)
    w = torch.eye(192, device=device)
    out = linear(x, w, None)
    torch.testing.assert_close(out, x, **TOL)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_linear.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement the kernel**

```python
# model/kernels/linear.py
import torch
import triton
import triton.language as tl
from torch import Tensor

from model.registry import Component, register


@triton.jit
def _linear_kernel(x_ptr, w_ptr, b_ptr, out_ptr,
                   M, N, K,
                   stride_xm, stride_xk,
                   stride_wn, stride_wk,
                   stride_om, stride_on,
                   HAS_BIAS: tl.constexpr,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                   BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    # weight is [N, K], so the N axis walks stride_wn and K walks stride_wk.
    w_ptrs = w_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        x = tl.load(x_ptrs,
                    mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_remaining),
                    other=0.0)
        w = tl.load(w_ptrs,
                    mask=(offs_n[None, :] < N) & (offs_k[:, None] < k_remaining),
                    other=0.0)
        acc += tl.dot(x, w)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    if HAS_BIAS:
        acc += tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)[None, :]

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@register(Component.LINEAR, "triton")
def linear(x: Tensor, weight: Tensor, bias: Tensor | None) -> Tensor:
    leading = x.shape[:-1]
    x_flat = x.contiguous().reshape(-1, x.shape[-1])
    m, k = x_flat.shape
    n = weight.shape[0]
    weight = weight.contiguous()
    out = torch.empty((m, n), device=x.device, dtype=x.dtype)

    grid = lambda meta: (triton.cdiv(m, meta["BLOCK_M"]),
                         triton.cdiv(n, meta["BLOCK_N"]))
    _linear_kernel[grid](
        x_flat, weight, bias if bias is not None else x_flat, out,
        m, n, k,
        x_flat.stride(0), x_flat.stride(1),
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1),
        HAS_BIAS=bias is not None,
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
        num_warps=4, num_stages=2)
    return out.reshape(*leading, n)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_linear.py -v`
Expected: PASS. `test_non_power_of_two_k_dimension` failing means the K-axis mask is wrong — the most common bug in this kernel.

- [ ] **Step 5: Benchmark against cuBLAS**

Create `bench/run_linear.py` sweeping the four model shapes and the batch axis. Use `bytes_theoretical = (batch*64*k + n*k + batch*64*n) * 4`.

Run: `python -m bench.run_linear`
Expected: **triton loses to cuBLAS**, likely by 2–5× without tensor cores. Record it. This baseline is what epilogue fusion in Task 12 must overcome, and the honest framing of the whole matmul story.

- [ ] **Step 6: Commit**

```bash
git add model/kernels/linear.py model/kernels/__init__.py tests/test_linear.py bench/run_linear.py bench/results/
git commit -m "feat(kernels): add Triton tiled linear"
```

---

### Task 11: Fused LayerNorm + residual

Rung 2 — the first real fusion. Hypothesis: `dram__bytes` drops by roughly a third.

**Files:**
- Modify: `model/kernels/layernorm.py`, `model/baseline/layers.py`
- Create: `tests/test_layernorm_residual.py`, `docs/findings/02-layernorm-fusion.md`

**Interfaces:**
- Consumes: existing `_layernorm_kernel` structure.
- Produces:
  - `layernorm_residual(x, residual, weight, bias, eps) -> tuple[Tensor, Tensor]` returning `(normed, updated_residual)` where `updated_residual = x + residual`, registered as `(LAYERNORM, "triton_residual")`.
  - Baseline twin `layernorm_residual` in `model/baseline/layers.py` registered as `(LAYERNORM, "torch_residual")`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_layernorm_residual.py
import torch
import torch.nn.functional as F
from model.kernels.layernorm import layernorm_residual
from tests.conftest import TOLERANCES

TOL = TOLERANCES["layernorm"]


def test_matches_unfused_sequence(device):
    x = torch.randn(8, 64, 192, device=device)
    residual = torch.randn(8, 64, 192, device=device)
    w = torch.randn(192, device=device)
    b = torch.randn(192, device=device)

    expected_residual = x + residual
    expected_normed = F.layer_norm(expected_residual, (192,), w, b, 1e-5)

    normed, updated = layernorm_residual(x, residual, w, b, 1e-5)
    torch.testing.assert_close(updated, expected_residual, **TOL)
    torch.testing.assert_close(normed, expected_normed, **TOL)


def test_batch_one(device):
    x = torch.randn(1, 64, 192, device=device)
    residual = torch.randn(1, 64, 192, device=device)
    w = torch.ones(192, device=device)
    b = torch.zeros(192, device=device)
    normed, updated = layernorm_residual(x, residual, w, b, 1e-5)
    torch.testing.assert_close(updated, x + residual, **TOL)
    torch.testing.assert_close(
        normed, F.layer_norm(x + residual, (192,), w, b, 1e-5), **TOL)


def test_residual_is_written_not_aliased(device):
    x = torch.randn(4, 64, 192, device=device)
    residual = torch.zeros(4, 64, 192, device=device)
    w = torch.ones(192, device=device)
    b = torch.zeros(192, device=device)
    _, updated = layernorm_residual(x, residual, w, b, 1e-5)
    assert updated.data_ptr() != residual.data_ptr()
    torch.testing.assert_close(updated, x, **TOL)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_layernorm_residual.py -v`
Expected: FAIL with `ImportError: cannot import name 'layernorm_residual'`

- [ ] **Step 3: Add the baseline twin**

```python
# append to model/baseline/layers.py
@register(Component.LAYERNORM, "torch_residual")
def layernorm_residual(x: Tensor, residual: Tensor, weight: Tensor,
                       bias: Tensor, eps: float) -> tuple[Tensor, Tensor]:
    updated = x + residual
    return layernorm(updated, weight, bias, eps), updated
```

- [ ] **Step 4: Implement the fused kernel**

```python
# append to model/kernels/layernorm.py
@triton.jit
def _layernorm_residual_kernel(x_ptr, res_ptr, w_ptr, b_ptr,
                               out_ptr, res_out_ptr,
                               stride_row, n_cols, eps,
                               BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offset = row * stride_row
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(x_ptr + offset + cols, mask=mask, other=0.0).to(tl.float32)
    res = tl.load(res_ptr + offset + cols, mask=mask, other=0.0).to(tl.float32)
    combined = x + res
    # Written out because the next block's residual path needs it; the
    # saving over the unfused pair is one read and one write of the
    # intermediate, not of this tensor.
    tl.store(res_out_ptr + offset + cols, combined, mask=mask)

    mean = tl.sum(combined, axis=0) / n_cols
    centered = tl.where(mask, combined - mean, 0.0)
    var = tl.sum(centered * centered, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(var + eps)

    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + offset + cols, centered * rstd * w + b, mask=mask)


@register(Component.LAYERNORM, "triton_residual")
def layernorm_residual(x: Tensor, residual: Tensor, weight: Tensor,
                       bias: Tensor, eps: float) -> tuple[Tensor, Tensor]:
    shape = x.shape
    x_flat = x.contiguous().reshape(-1, shape[-1])
    res_flat = residual.contiguous().reshape(-1, shape[-1])
    rows, n_cols = x_flat.shape
    out = torch.empty_like(x_flat)
    res_out = torch.empty_like(x_flat)
    block = triton.next_power_of_2(n_cols)
    _layernorm_residual_kernel[(rows,)](
        x_flat, res_flat, weight, bias, out, res_out,
        x_flat.stride(0), n_cols, eps,
        BLOCK=block, num_warps=4 if block <= 512 else 8)
    return out.reshape(shape), res_out.reshape(shape)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_layernorm_residual.py -v`
Expected: PASS

- [ ] **Step 6: Benchmark and profile the fusion**

Create `bench/run_layernorm_residual.py` comparing three arms: `torch` (separate add then layer_norm), `triton` (separate), `triton_residual` (fused). Then profile the unfused and fused Triton arms.

Run: `python -m bench.run_layernorm_residual`
Then collect counters for both variants at batch 128 via `profile_kernel`.

Expected: `dram__bytes_read.sum + dram__bytes_write.sum` measurably lower for the fused variant. Compute the exact ratio and compare it to the predicted ⅓ reduction.

- [ ] **Step 7: Write the findings document**

Create `docs/findings/02-layernorm-fusion.md` with the latency table, the DRAM byte counts for both variants, the measured versus predicted traffic reduction, and an explanation of any discrepancy. If measured traffic did not fall as predicted, state why — L2 may already have been absorbing the intermediate, which is itself the finding.

- [ ] **Step 8: Commit**

```bash
git add model/kernels/layernorm.py model/baseline/layers.py tests/test_layernorm_residual.py bench/run_layernorm_residual.py docs/findings/02-layernorm-fusion.md bench/results/
git commit -m "feat(kernels): fuse residual add into LayerNorm"
```

---

### Task 12: Linear with bias and GeLU epilogue

Rungs 6 and 7. Eliminates a full round-trip of the `[B, 64, 768]` intermediate.

**Files:**
- Modify: `model/kernels/linear.py`, `model/baseline/layers.py`
- Create: `tests/test_linear_gelu.py`, `docs/findings/03-epilogue-fusion.md`

**Interfaces:**
- Consumes: `_linear_kernel` from Task 10.
- Produces:
  - `linear_gelu(x, weight, bias) -> Tensor` registered as `(LINEAR, "triton_gelu")` — computes `gelu(x @ weight.T + bias)` in one kernel.
  - Baseline twin registered as `(LINEAR, "torch_gelu")`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_linear_gelu.py
import pytest
import torch
import torch.nn.functional as F
from model.kernels.linear import linear_gelu
from tests.conftest import TOLERANCES

TOL = TOLERANCES["linear"]


@pytest.mark.parametrize("batch", [1, 8, 128])
def test_matches_unfused_sequence(device, batch):
    x = torch.randn(batch, 64, 192, device=device)
    w = torch.randn(768, 192, device=device) * 0.05
    b = torch.randn(768, device=device)
    expected = F.gelu(F.linear(x, w, b), approximate="tanh")
    torch.testing.assert_close(linear_gelu(x, w, b), expected, **TOL)


def test_second_mlp_shape(device):
    x = torch.randn(4, 64, 768, device=device)
    w = torch.randn(192, 768, device=device) * 0.05
    b = torch.randn(192, device=device)
    expected = F.gelu(F.linear(x, w, b), approximate="tanh")
    torch.testing.assert_close(linear_gelu(x, w, b), expected, **TOL)


def test_large_activations_do_not_overflow(device):
    x = torch.full((2, 64, 192), 10.0, device=device)
    w = torch.full((768, 192), 0.5, device=device)
    b = torch.zeros(768, device=device)
    out = linear_gelu(x, w, b)
    assert torch.isfinite(out).all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_linear_gelu.py -v`
Expected: FAIL with `ImportError: cannot import name 'linear_gelu'`

- [ ] **Step 3: Add the baseline twin**

```python
# append to model/baseline/layers.py
@register(Component.LINEAR, "torch_gelu")
def linear_gelu(x: Tensor, weight: Tensor, bias: Tensor | None) -> Tensor:
    return gelu(linear(x, weight, bias))
```

- [ ] **Step 4: Implement the fused kernel**

```python
# append to model/kernels/linear.py
@triton.jit
def _linear_gelu_kernel(x_ptr, w_ptr, b_ptr, out_ptr,
                        M, N, K,
                        stride_xm, stride_xk,
                        stride_wn, stride_wk,
                        stride_om, stride_on,
                        HAS_BIAS: tl.constexpr,
                        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                        BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = w_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        x = tl.load(x_ptrs,
                    mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_remaining),
                    other=0.0)
        w = tl.load(w_ptrs,
                    mask=(offs_n[None, :] < N) & (offs_k[:, None] < k_remaining),
                    other=0.0)
        acc += tl.dot(x, w)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    if HAS_BIAS:
        acc += tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)[None, :]

    # The epilogue: the [M, N] tile never leaves registers between the
    # matmul and the activation.
    inner = 0.7978845608028654 * (acc + 0.044715 * acc * acc * acc)
    acc = 0.5 * acc * (1.0 + tl.math.tanh(inner))

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@register(Component.LINEAR, "triton_gelu")
def linear_gelu(x: Tensor, weight: Tensor, bias: Tensor | None) -> Tensor:
    leading = x.shape[:-1]
    x_flat = x.contiguous().reshape(-1, x.shape[-1])
    m, k = x_flat.shape
    n = weight.shape[0]
    weight = weight.contiguous()
    out = torch.empty((m, n), device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(m, meta["BLOCK_M"]),
                         triton.cdiv(n, meta["BLOCK_N"]))
    _linear_gelu_kernel[grid](
        x_flat, weight, bias if bias is not None else x_flat, out,
        m, n, k,
        x_flat.stride(0), x_flat.stride(1),
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1),
        HAS_BIAS=bias is not None,
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
        num_warps=4, num_stages=2)
    return out.reshape(*leading, n)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_linear_gelu.py -v`
Expected: PASS

- [ ] **Step 6: Benchmark and profile**

Create `bench/run_linear_gelu.py` with arms `torch_gelu`, `triton` + separate `gelu`, and `triton_gelu`. Profile the last two at batch 128.

Expected: fused variant saves roughly `2 * batch * 64 * 768 * 4` bytes of traffic. Check `launch__registers_per_thread` — if the epilogue pushed register usage up, note the value; it is the first data point on the register-pressure curve that Tasks 16 and 17 test.

- [ ] **Step 7: Write findings**

Create `docs/findings/03-epilogue-fusion.md` covering both fused matmul variants: latency versus cuBLAS, traffic saved, register delta, and whether fusion closed any of the gap to cuBLAS measured in Task 10.

- [ ] **Step 8: Commit**

```bash
git add model/kernels/linear.py model/baseline/layers.py tests/test_linear_gelu.py bench/run_linear_gelu.py docs/findings/03-epilogue-fusion.md bench/results/
git commit -m "feat(kernels): fuse bias and GeLU epilogue into linear"
```

---

### Task 13: Composed Triton attention

Rung 8. Five launches and a materialized `[B, H, 64, 64]` score matrix — the baseline that rungs 9 and 10 must beat.

**Files:**
- Create: `model/kernels/attention.py`, `tests/test_attention.py`
- Modify: `model/kernels/__init__.py`

**Interfaces:**
- Consumes: `linear` (Task 10), `softmax` (Task 9).
- Produces: `attention_composed(q, k, v, scale) -> Tensor` registered as `(ATTENTION, "triton_composed")`. Inputs `[B, H, S, Dh]`, output `[B, H, S, Dh]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_attention.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_attention.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement composed attention**

```python
# model/kernels/attention.py
import torch
from torch import Tensor

from model.kernels.softmax import softmax
from model.registry import Component, register


@register(Component.ATTENTION, "triton_composed")
def attention_composed(q: Tensor, k: Tensor, v: Tensor, scale: float) -> Tensor:
    """Rung 8: unfused. The [B, H, S, S] score matrix is materialized in
    DRAM between the two matmuls -- this is exactly what rung 10 removes."""
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    return torch.matmul(softmax(scores), v)
```

Add `from model.kernels import attention  # noqa: F401` to `model/kernels/__init__.py`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_attention.py -v`
Expected: PASS

- [ ] **Step 5: Benchmark and record the materialized-score traffic**

Create `bench/run_attention.py`. For this variant `bytes_theoretical` must include the score matrix round-trip: `(3 * batch*3*64*64 + 2 * batch*3*64*64 + batch*3*64*64) * 4`. Profile at batch 128 and record `dram__bytes_read.sum + dram__bytes_write.sum` — this number is the target rung 10 attacks.

- [ ] **Step 6: Commit**

```bash
git add model/kernels/attention.py model/kernels/__init__.py tests/test_attention.py bench/run_attention.py bench/results/
git commit -m "feat(kernels): add composed Triton attention"
```

---

### Task 14: Fused QKV projection

Rung 9. Three GEMMs become one, raising arithmetic intensity and cutting two launches.

**Files:**
- Modify: `model/kernels/attention.py`
- Create: `tests/test_qkv.py`

**Interfaces:**
- Consumes: `linear` from Task 10.
- Produces: `qkv_project(x, qkv_w, qkv_b, heads) -> tuple[Tensor, Tensor, Tensor]` — `x` is `[B, S, D]`, `qkv_w` is `[3D, D]`; returns `q`, `k`, `v` each `[B, heads, S, head_dim]`. Registered as `(ATTENTION, "triton_qkv_fused")` via a wrapper `attention_qkv_fused(x, qkv_w, qkv_b, heads, scale) -> Tensor`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_qkv.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_qkv.py -v`
Expected: FAIL with `ImportError: cannot import name 'qkv_project'`

- [ ] **Step 3: Implement fused QKV projection**

```python
# append to model/kernels/attention.py
from model.kernels.linear import linear


def qkv_project(x: Tensor, qkv_w: Tensor, qkv_b: Tensor,
                heads: int) -> tuple[Tensor, Tensor, Tensor]:
    """Rung 9: one [D -> 3D] GEMM instead of three [D -> D] GEMMs.

    The weight layout must match the reference exactly: rows [0:D] are Q,
    [D:2D] are K, [2D:3D] are V. Getting this wrong produces a model that
    trains fine but loads the checkpoint incorrectly.
    """
    batch, seq, dim = x.shape
    head_dim = dim // heads
    packed = linear(x, qkv_w, qkv_b)
    packed = packed.reshape(batch, seq, 3, heads, head_dim)
    packed = packed.permute(2, 0, 3, 1, 4)
    return packed[0], packed[1], packed[2]


@register(Component.ATTENTION, "triton_qkv_fused")
def attention_qkv_fused(x: Tensor, qkv_w: Tensor, qkv_b: Tensor,
                        heads: int, scale: float) -> Tensor:
    q, k, v = qkv_project(x, qkv_w, qkv_b, heads)
    batch, seq = x.shape[0], x.shape[1]
    out = attention_composed(q, k, v, scale)
    return out.transpose(1, 2).reshape(batch, seq, -1)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_qkv.py -v`
Expected: PASS

- [ ] **Step 5: Benchmark launch count with nsys**

Run: `nsys profile --stats=true -o bench/results/nsys_qkv python -m bench.run_attention`
Expected: the CUDA kernel summary shows fewer launches for the fused arm. Record the launch counts for both arms and the latency delta at batch 1, where launch overhead dominates.

- [ ] **Step 6: Commit**

```bash
git add model/kernels/attention.py tests/test_qkv.py bench/results/
git commit -m "feat(kernels): fuse QKV into a single projection GEMM"
```

---

### Task 15: Flash-style fused attention

Rung 10, the headline kernel. At S=64 and head_dim=64 an entire head's Q, K, V is 48KB in fp32 — inside Turing's 64KB shared memory — so the tiling loop collapses to a single block and the score matrix never reaches DRAM.

**Files:**
- Modify: `model/kernels/attention.py`, `tests/test_attention.py`
- Create: `docs/findings/04-flash-attention.md`

**Interfaces:**
- Consumes: nothing new.
- Produces: `attention_flash(q, k, v, scale) -> Tensor` registered as `(ATTENTION, "triton_flash")`. Same signature as `attention_composed`.

- [ ] **Step 1: Add the new variant to the shared attention test list**

Modify `tests/test_attention.py`:

```python
from model.kernels.attention import attention_composed, attention_flash

VARIANTS = [attention_composed, attention_flash]
```

All five existing parametrized tests now run against the flash variant too. Add one more:

```python
def test_flash_never_materializes_scores(device):
    """A [B,H,S,S] score buffer for batch 512 would be 400MB. If the kernel
    allocates one, this OOMs or shows a large allocation spike."""
    torch.cuda.reset_peak_memory_stats()
    q, k, v = (torch.randn(256, 3, 64, 64, device=device) for _ in range(3))
    baseline = torch.cuda.max_memory_allocated()
    attention_flash(q, k, v, 64 ** -0.5)
    peak = torch.cuda.max_memory_allocated()
    score_bytes = 256 * 3 * 64 * 64 * 4
    assert peak - baseline < score_bytes
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_attention.py -v`
Expected: FAIL with `ImportError: cannot import name 'attention_flash'`

- [ ] **Step 3: Implement the fused kernel**

```python
# append to model/kernels/attention.py
import triton
import triton.language as tl


@triton.jit
def _flash_kernel(q_ptr, k_ptr, v_ptr, out_ptr,
                  stride_qb, stride_qh, stride_qs, stride_qd,
                  stride_ob, stride_oh, stride_os, stride_od,
                  seq_len, scale,
                  BLOCK_S: tl.constexpr, BLOCK_D: tl.constexpr):
    """One program per (batch, head). At S=64, Dh=64 the whole head fits in
    SRAM, so there is no outer tile loop and online rescaling degenerates
    to a single softmax pass. This is a property of the small problem size,
    not a general FlashAttention implementation.
    """
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_s = tl.arange(0, BLOCK_S)
    offs_d = tl.arange(0, BLOCK_D)
    mask_s = offs_s < seq_len

    base = pid_b * stride_qb + pid_h * stride_qh
    qkv_offsets = base + offs_s[:, None] * stride_qs + offs_d[None, :] * stride_qd
    load_mask = mask_s[:, None]

    q = tl.load(q_ptr + qkv_offsets, mask=load_mask, other=0.0).to(tl.float32)
    k = tl.load(k_ptr + qkv_offsets, mask=load_mask, other=0.0).to(tl.float32)
    v = tl.load(v_ptr + qkv_offsets, mask=load_mask, other=0.0).to(tl.float32)

    scores = tl.dot(q, tl.trans(k)) * scale
    # Masked key positions must lose the row max and contribute zero weight.
    scores = tl.where(mask_s[None, :], scores, float("-inf"))
    scores = scores - tl.max(scores, axis=1)[:, None]
    weights = tl.exp(scores)
    weights = tl.where(mask_s[None, :], weights, 0.0)
    weights = weights / tl.sum(weights, axis=1)[:, None]

    out = tl.dot(weights.to(v.dtype), v)
    out_offsets = (pid_b * stride_ob + pid_h * stride_oh
                   + offs_s[:, None] * stride_os + offs_d[None, :] * stride_od)
    tl.store(out_ptr + out_offsets, out, mask=load_mask)


@register(Component.ATTENTION, "triton_flash")
def attention_flash(q: Tensor, k: Tensor, v: Tensor, scale: float) -> Tensor:
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    batch, heads, seq_len, head_dim = q.shape
    out = torch.empty_like(q)
    _flash_kernel[(batch, heads)](
        q, k, v, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        seq_len, scale,
        BLOCK_S=triton.next_power_of_2(seq_len),
        BLOCK_D=triton.next_power_of_2(head_dim),
        num_warps=4, num_stages=2)
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_attention.py -v`
Expected: PASS for both variants. If `test_large_scores_do_not_overflow` fails only for flash, the `-inf` masking interacts badly with the row max — check that at least one lane per row is unmasked.

- [ ] **Step 5: Benchmark and profile against the composed variant**

Run: `python -m bench.run_attention` with all three arms (`torch`, `triton_composed`, `triton_flash`), then profile the composed and flash variants at batch 128.

Expected: the largest single win in the project. DRAM traffic should fall by the full score-matrix round-trip recorded in Task 13. **Check achieved occupancy** — 48KB of shared memory per block permits one resident block per SM, so `sm__warps_active` may be low even as latency improves. That tension is the most interesting result here.

- [ ] **Step 6: Write findings**

Create `docs/findings/04-flash-attention.md`: latency across the batch sweep for all three arms, DRAM traffic before and after, achieved occupancy, and an explicit discussion of whether occupancy of 1 hurt on a 16-SM card. Compare against the Modal GPU if available.

- [ ] **Step 7: Commit**

```bash
git add model/kernels/attention.py tests/test_attention.py docs/findings/04-flash-attention.md bench/results/
git commit -m "feat(kernels): add flash-style fused attention"
```

---

### Task 16: Whole-MLP mega-kernel

Rung 12. **This is predicted to hurt.** The 768-wide intermediate must live in registers across both matmuls; the hypothesis is that it spills to local memory and occupancy collapses.

**Files:**
- Create: `model/kernels/mlp.py`, `tests/test_mlp.py`, `docs/findings/05-over-fusion.md`
- Modify: `model/kernels/__init__.py`

**Interfaces:**
- Consumes: `linear`, `linear_gelu` from Tasks 10 and 12.
- Produces:
  - `mlp_composed(x, w1, b1, w2, b2) -> Tensor` registered as `(MLP, "triton_composed")` — `linear_gelu` then `linear`, two kernels.
  - `mlp_fused(x, w1, b1, w2, b2) -> Tensor` registered as `(MLP, "triton_fused")` — one kernel, hidden dimension never leaves the SM.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_mlp.py
import pytest
import torch
import torch.nn.functional as F
from model.kernels.mlp import mlp_composed, mlp_fused
from tests.conftest import TOLERANCES

TOL = TOLERANCES["mlp"]
VARIANTS = [mlp_composed, mlp_fused]


@pytest.mark.parametrize("fn", VARIANTS)
@pytest.mark.parametrize("batch", [1, 8, 128])
def test_matches_torch(device, fn, batch):
    x = torch.randn(batch, 64, 192, device=device)
    w1 = torch.randn(768, 192, device=device) * 0.05
    b1 = torch.randn(768, device=device)
    w2 = torch.randn(192, 768, device=device) * 0.05
    b2 = torch.randn(192, device=device)
    expected = F.linear(F.gelu(F.linear(x, w1, b1), approximate="tanh"), w2, b2)
    torch.testing.assert_close(fn(x, w1, b1, w2, b2), expected, **TOL)


@pytest.mark.parametrize("fn", VARIANTS)
def test_preserves_shape(device, fn):
    x = torch.randn(4, 64, 192, device=device)
    w1 = torch.randn(768, 192, device=device) * 0.05
    b1 = torch.zeros(768, device=device)
    w2 = torch.randn(192, 768, device=device) * 0.05
    b2 = torch.zeros(192, device=device)
    assert fn(x, w1, b1, w2, b2).shape == x.shape


def test_fused_matches_composed_exactly_enough(device):
    """Both are Triton and share accumulation strategy, so they should agree
    more tightly than either agrees with torch."""
    x = torch.randn(8, 64, 192, device=device)
    w1 = torch.randn(768, 192, device=device) * 0.05
    b1 = torch.randn(768, device=device)
    w2 = torch.randn(192, 768, device=device) * 0.05
    b2 = torch.randn(192, device=device)
    torch.testing.assert_close(mlp_fused(x, w1, b1, w2, b2),
                               mlp_composed(x, w1, b1, w2, b2),
                               rtol=1e-5, atol=1e-5)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_mlp.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement both variants**

```python
# model/kernels/mlp.py
import torch
import triton
import triton.language as tl
from torch import Tensor

from model.kernels.linear import linear, linear_gelu
from model.registry import Component, register


@register(Component.MLP, "triton_composed")
def mlp_composed(x: Tensor, w1: Tensor, b1: Tensor,
                 w2: Tensor, b2: Tensor) -> Tensor:
    return linear(linear_gelu(x, w1, b1), w2, b2)


@triton.jit
def _mlp_fused_kernel(x_ptr, w1_ptr, b1_ptr, w2_ptr, b2_ptr, out_ptr,
                      M, D, H,
                      stride_xm, stride_xk,
                      stride_w1n, stride_w1k,
                      stride_w2n, stride_w2k,
                      stride_om, stride_on,
                      BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr,
                      BLOCK_H: tl.constexpr):
    """Rung 12: the deliberate over-fusion.

    Each program owns a [BLOCK_M, D] output tile and must hold the entire
    [BLOCK_M, H] hidden activation to produce it, because the second matmul
    reduces over H. With H=768 that is 64*768 floats per program -- far more
    than the register file holds, so this is expected to spill to local
    memory. The spill is the finding; check the local_op_ld/st counters.
    """
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    offs_h = tl.arange(0, BLOCK_H)

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_d[None, :] * stride_xk
    x = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & (offs_d[None, :] < D),
                other=0.0)

    w1_ptrs = w1_ptr + offs_h[None, :] * stride_w1n + offs_d[:, None] * stride_w1k
    w1 = tl.load(w1_ptrs, mask=(offs_h[None, :] < H) & (offs_d[:, None] < D),
                 other=0.0)

    hidden = tl.dot(x, w1)
    hidden += tl.load(b1_ptr + offs_h, mask=offs_h < H, other=0.0)[None, :]
    inner = 0.7978845608028654 * (hidden + 0.044715 * hidden * hidden * hidden)
    hidden = 0.5 * hidden * (1.0 + tl.math.tanh(inner))
    hidden = tl.where(offs_h[None, :] < H, hidden, 0.0)

    w2_ptrs = w2_ptr + offs_d[None, :] * stride_w2n + offs_h[:, None] * stride_w2k
    w2 = tl.load(w2_ptrs, mask=(offs_d[None, :] < D) & (offs_h[:, None] < H),
                 other=0.0)

    out = tl.dot(hidden.to(w2.dtype), w2)
    out += tl.load(b2_ptr + offs_d, mask=offs_d < D, other=0.0)[None, :]

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_d[None, :] * stride_on
    tl.store(out_ptrs, out, mask=(offs_m[:, None] < M) & (offs_d[None, :] < D))


@register(Component.MLP, "triton_fused")
def mlp_fused(x: Tensor, w1: Tensor, b1: Tensor,
              w2: Tensor, b2: Tensor) -> Tensor:
    shape = x.shape
    x_flat = x.contiguous().reshape(-1, shape[-1])
    m, dim = x_flat.shape
    hidden_dim = w1.shape[0]
    w1, w2 = w1.contiguous(), w2.contiguous()
    out = torch.empty_like(x_flat)
    grid = lambda meta: (triton.cdiv(m, meta["BLOCK_M"]),)
    _mlp_fused_kernel[grid](
        x_flat, w1, b1, w2, b2, out,
        m, dim, hidden_dim,
        x_flat.stride(0), x_flat.stride(1),
        w1.stride(0), w1.stride(1),
        w2.stride(0), w2.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=32,
        BLOCK_D=triton.next_power_of_2(dim),
        BLOCK_H=triton.next_power_of_2(hidden_dim),
        num_warps=8, num_stages=1)
    return out.reshape(shape)
```

Add `from model.kernels import mlp  # noqa: F401` to `model/kernels/__init__.py`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_mlp.py -v`
Expected: PASS. Two plausible failure modes, both findings rather than defects to hide:
- **Out of shared memory / resource errors at compile time.** Reduce `BLOCK_M` to 16, then 8. Record the largest `BLOCK_M` that compiles — that ceiling is itself the over-fusion result.
- **Correctness failures only at large batch.** Check the `M` mask on the store.

- [ ] **Step 5: Benchmark and profile the over-fusion**

Run: `python -m bench.run_mlp` with arms `torch`, `triton_composed`, `triton_fused` across the full batch sweep. Profile both Triton arms at batch 1 and batch 512 — both ends, because the hypothesis is that the fused variant wins at batch 1 and loses at batch 512.

Expected signature for the fused variant relative to composed:
- `dram__bytes_*` **down** (the hidden activation never reaches DRAM)
- `launch__registers_per_thread` **up**
- `sm__warps_active...` **down**
- `l1tex__t_bytes_pipe_lsu_mem_local_op_ld/st.sum` **non-zero** — direct evidence of spilling

- [ ] **Step 6: Write the over-fusion findings**

Create `docs/findings/05-over-fusion.md`. This is the project's central document. Record all four counters for both variants at both batch extremes, the latency crossover point, and a direct answer to: *at what batch size does fusion stop paying, and which counter explains it?* If the fused variant never loses, say so plainly and explain what that implies about the register file versus the memory system on this card.

- [ ] **Step 7: Commit**

```bash
git add model/kernels/mlp.py model/kernels/__init__.py tests/test_mlp.py bench/run_mlp.py docs/findings/05-over-fusion.md bench/results/
git commit -m "feat(kernels): add fused whole-MLP kernel for over-fusion study"
```

---

### Task 17: Fully fused transformer block

Rung 13. Predicted to hurt more than Task 16 — this is the deliberate far end of the ladder.

**Files:**
- Create: `model/kernels/block.py`, `tests/test_block.py`
- Modify: `model/kernels/__init__.py`, `docs/findings/05-over-fusion.md`

**Interfaces:**
- Consumes: kernels from Tasks 11, 12, 14, 15, 16.
- Produces:
  - `block_composed(...) -> Tensor` registered as `(BLOCK, "triton_composed")` — same signature as the baseline `block`, assembled from the best individual Triton variants (`layernorm_residual`, `qkv_project`, `attention_flash`, `linear`, `mlp_composed`).
  - `block_fused(...) -> Tensor` registered as `(BLOCK, "triton_fused")` — same signature; attention and MLP in one kernel where the shared-memory budget allows, otherwise the minimum viable number of launches, with the actual achieved fusion recorded.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_block.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_block.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement the composed block**

```python
# model/kernels/block.py
from torch import Tensor

from model.kernels.attention import attention_flash, qkv_project
from model.kernels.layernorm import layernorm, layernorm_residual
from model.kernels.linear import linear
from model.kernels.mlp import mlp_composed, mlp_fused
from model.registry import Component, register


def _block(x, ln1_w, ln1_b, qkv_w, qkv_b, proj_w, proj_b, ln2_w, ln2_b,
           w1, b1, w2, b2, heads, scale, eps, mlp_fn):
    batch, seq, dim = x.shape
    normed = layernorm(x, ln1_w, ln1_b, eps)
    q, k, v = qkv_project(normed, qkv_w, qkv_b, heads)
    attended = attention_flash(q, k, v, scale)
    attended = attended.transpose(1, 2).reshape(batch, seq, dim)
    normed, residual = layernorm_residual(
        linear(attended, proj_w, proj_b), x, ln2_w, ln2_b, eps)
    return residual + mlp_fn(normed, w1, b1, w2, b2)


@register(Component.BLOCK, "triton_composed")
def block_composed(x: Tensor, ln1_w, ln1_b, qkv_w, qkv_b, proj_w, proj_b,
                   ln2_w, ln2_b, w1, b1, w2, b2,
                   heads: int, scale: float, eps: float) -> Tensor:
    return _block(x, ln1_w, ln1_b, qkv_w, qkv_b, proj_w, proj_b, ln2_w, ln2_b,
                  w1, b1, w2, b2, heads, scale, eps, mlp_composed)


@register(Component.BLOCK, "triton_fused")
def block_fused(x: Tensor, ln1_w, ln1_b, qkv_w, qkv_b, proj_w, proj_b,
                ln2_w, ln2_b, w1, b1, w2, b2,
                heads: int, scale: float, eps: float) -> Tensor:
    """Rung 13: maximum fusion. Uses the mega-MLP from rung 12 on top of the
    fused attention and fused LayerNorm+residual, minimizing launch count at
    the cost of the highest register pressure in the project."""
    return _block(x, ln1_w, ln1_b, qkv_w, qkv_b, proj_w, proj_b, ln2_w, ln2_b,
                  w1, b1, w2, b2, heads, scale, eps, mlp_fused)
```

Add `from model.kernels import block  # noqa: F401` to `model/kernels/__init__.py`.

**Note on how far to fuse:** merging attention and the MLP into a single Triton kernel requires one program to hold a `[BLOCK_M, 768]` hidden tile *and* a `[64, 64]` attention tile simultaneously. If that compiles at any `BLOCK_M`, implement it as an additional variant `triton_fused_monolithic` and measure it. If it fails to compile, **record the failure and its error message in the findings document** — a fusion the compiler refuses is a legitimate and informative upper bound on the ladder, and the composition above stands as the maximum achievable rung.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_block.py -v`
Expected: PASS. The `block` tolerance is looser (`rtol=1e-3`) because errors compound across six sub-operations. If it fails by more than 10×, the cause is a real bug — most likely the QKV weight-slice ordering or a residual applied to the wrong tensor.

- [ ] **Step 5: Benchmark and profile with both tools**

Run: `python -m bench.run_block` with arms `torch`, `triton_composed`, `triton_fused` across the batch sweep.
Then: `nsys profile --stats=true -o bench/results/nsys_block python -m bench.run_block` to capture launch counts per arm.
Then: `profile_kernel` on both Triton arms at batch 1 and batch 512.

Expected: `triton_fused` has the fewest launches and wins at batch 1; `triton_composed` wins at batch 512. That crossover, if it appears, is the project's headline result.

- [ ] **Step 6: Extend the over-fusion findings**

Append to `docs/findings/05-over-fusion.md`: block-level launch counts from `nsys`, counters from `ncu`, the latency crossover batch, and whether the monolithic variant compiled. State the conclusion in one sentence — the condition under which fusion stops paying on this hardware.

- [ ] **Step 7: Commit**

```bash
git add model/kernels/block.py model/kernels/__init__.py tests/test_block.py bench/run_block.py docs/findings/05-over-fusion.md bench/results/
git commit -m "feat(kernels): add fully fused transformer block"
```

---

### Task 18: Backend-switchable ViT and end-to-end accuracy gate

**Files:**
- Create: `model/vit.py`, `tests/test_end_to_end.py`

**Interfaces:**
- Consumes: `VariantConfig`, `ViTConfig`, baseline `VisionTransformer`, all registered variants.
- Produces: `TritonViT(cfg: ViTConfig, variants: VariantConfig)` — `nn.Module` sharing the baseline's parameter names so `load_state_dict` accepts the frozen checkpoint directly; `forward(images) -> Tensor`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_end_to_end.py
import pytest
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model.baseline.vit import VisionTransformer
from model.config import ViTConfig
from model.registry import VariantConfig
from model.vit import TritonViT

CHECKPOINT = "data/checkpoint.pt"
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def load_models(device, variants):
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    cfg = ViTConfig()
    reference = VisionTransformer(cfg).to(device).eval()
    reference.load_state_dict(ckpt["state_dict"])
    triton_model = TritonViT(cfg, variants).to(device).eval()
    triton_model.load_state_dict(ckpt["state_dict"])
    return reference, triton_model, ckpt["test_accuracy"]


def test_logits_match_reference_on_random_input(device):
    variants = VariantConfig(block="triton_composed")
    reference, triton_model, _ = load_models(device, variants)
    images = torch.randn(16, 3, 32, 32, device=device)
    with torch.no_grad():
        torch.testing.assert_close(triton_model(images), reference(images),
                                   rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("block_variant", ["triton_composed", "triton_fused"])
def test_cifar10_accuracy_and_prediction_agreement(device, block_variant):
    """Accuracy alone is too coarse: two models can score identically while
    disagreeing on many images. Both gates must hold."""
    variants = VariantConfig(block=block_variant)
    reference, triton_model, recorded = load_models(device, variants)

    test_tf = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize(MEAN, STD)])
    loader = DataLoader(datasets.CIFAR10("data", train=False, download=True,
                                         transform=test_tf),
                        batch_size=256, shuffle=False)

    correct = agree = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            ref_pred = reference(images).argmax(-1)
            tri_pred = triton_model(images).argmax(-1)
            correct += (tri_pred.cpu() == labels).sum().item()
            agree += (tri_pred == ref_pred).sum().item()
            total += labels.numel()

    assert abs(correct / total - recorded) < 0.001
    assert agree / total >= 0.999
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_end_to_end.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'model.vit'`

- [ ] **Step 3: Implement the switchable model**

```python
# model/vit.py
import torch
from torch import Tensor, nn

import model.baseline  # noqa: F401  (registers torch variants)
import model.kernels   # noqa: F401  (registers triton variants)
from model.baseline.vit import BlockParams
from model.config import ViTConfig
from model.registry import Component, VariantConfig


class TritonViT(nn.Module):
    """Parameter names match the baseline exactly so the frozen checkpoint
    loads without translation. Only the compute path differs."""

    def __init__(self, cfg: ViTConfig, variants: VariantConfig | None = None):
        super().__init__()
        self.cfg = cfg
        self.variants = variants or VariantConfig()
        self.block_fn = self.variants.resolve(Component.BLOCK)
        self.layernorm_fn = self.variants.resolve(Component.LAYERNORM)
        self.linear_fn = self.variants.resolve(Component.LINEAR)

        patch_elems = cfg.in_channels * cfg.patch_size ** 2
        self.patch_w = nn.Parameter(torch.empty(cfg.dim, patch_elems))
        self.patch_b = nn.Parameter(torch.zeros(cfg.dim))
        self.pos = nn.Parameter(torch.zeros(1, cfg.num_patches, cfg.dim))
        self.blocks = nn.ModuleList(BlockParams(cfg) for _ in range(cfg.depth))
        self.norm_w = nn.Parameter(torch.ones(cfg.dim))
        self.norm_b = nn.Parameter(torch.zeros(cfg.dim))
        self.head_w = nn.Parameter(torch.empty(cfg.num_classes, cfg.dim))
        self.head_b = nn.Parameter(torch.zeros(cfg.num_classes))

    def embed(self, images: Tensor) -> Tensor:
        cfg = self.cfg
        patches = images.unfold(2, cfg.patch_size, cfg.patch_size) \
                        .unfold(3, cfg.patch_size, cfg.patch_size)
        patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(
            images.shape[0], cfg.num_patches, -1)
        return self.linear_fn(patches, self.patch_w, self.patch_b) + self.pos

    def forward(self, images: Tensor) -> Tensor:
        cfg = self.cfg
        x = self.embed(images)
        for blk in self.blocks:
            x = self.block_fn(x, blk.ln1_w, blk.ln1_b, blk.qkv_w, blk.qkv_b,
                              blk.proj_w, blk.proj_b, blk.ln2_w, blk.ln2_b,
                              blk.w1, blk.b1, blk.w2, blk.b2,
                              cfg.heads, cfg.scale, cfg.eps)
        x = self.layernorm_fn(x, self.norm_w, self.norm_b, cfg.eps)
        return self.linear_fn(x.mean(dim=1), self.head_w, self.head_b)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_end_to_end.py -v`
Expected: PASS. If accuracy matches but agreement falls below 99.9%, a kernel is subtly wrong in a way that averages out — bisect by setting `block="torch"` and promoting one component at a time.

- [ ] **Step 5: Run the full suite**

Run: `pytest tests/ -v`
Expected: all PASS. This is the first point at which every kernel is exercised together.

- [ ] **Step 6: Commit**

```bash
git add model/vit.py tests/test_end_to_end.py
git commit -m "feat(model): add backend-switchable ViT with accuracy gate"
```

---

### Task 19: Full sweep and synthesis

Produces the project's actual output: the complete measurement grid and the written answer to the research question.

**Files:**
- Create: `bench/run_sweep.py`, `docs/findings/06-synthesis.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: everything.
- Produces: `bench/results/latency.csv` and `bench/results/counters.csv` covering the full grid, plus the synthesis document.

- [ ] **Step 1: Write the sweep driver**

```python
# bench/run_sweep.py
"""Full measurement grid: every registered variant across the batch sweep.

Truncates the batch sweep on OOM rather than failing, recording the
achieved ceiling -- 4GB is not enough for every variant at batch 512.
"""
import argparse
import torch

from bench.clocks import locked_clock_mhz
from bench.harness import Measurement, compare, record
from model.config import ViTConfig
from model.registry import Component, VariantConfig, variants
from model.vit import TritonViT

BATCHES = [1, 8, 32, 128, 512]


def block_arms(cfg, device):
    models = {}
    for variant in variants(Component.BLOCK):
        try:
            models[variant] = TritonViT(
                cfg, VariantConfig(block=variant)).to(device).eval()
        except ValueError:
            continue
    return models


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="bench/results/latency.csv")
    parser.add_argument("--reps", type=int, default=30)
    args = parser.parse_args()

    device = torch.device("cuda")
    cfg = ViTConfig()
    models = block_arms(cfg, device)
    locked = locked_clock_mhz()
    rows = []

    for batch in BATCHES:
        try:
            images = torch.randn(batch, 3, 32, 32, device=device)
            arms = {}
            for name, model in models.items():
                arms[name] = (lambda m=model, im=images:
                              torch.no_grad().__enter__() or m(im))
            samples = compare(arms, reps=args.reps)
        except torch.cuda.OutOfMemoryError:
            print(f"batch {batch}: OOM -- sweep ceiling reached")
            torch.cuda.empty_cache()
            break

        bytes_theoretical = batch * 3 * 32 * 32 * 4
        for name, values in samples.items():
            rows.append(Measurement.build(
                kernel="vit_forward", variant=name, batch=batch,
                dtype="float32", samples=values,
                bytes_theoretical=bytes_theoretical,
                locked_clock_mhz=locked))
        print(f"batch {batch}: " + ", ".join(
            f"{n}={min(v):.3f}ms" for n, v in samples.items()))

    record(rows, args.out)
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Lock clocks and run the sweep**

Run:
```bash
bash scripts/lock_clocks.sh 1200
python -m bench.run_sweep
```
Expected: a latency table per batch size, and an OOM-truncated ceiling recorded if batch 512 does not fit in 4GB.

- [ ] **Step 3: Check for thermal contamination**

Run: `python -c "import csv; rows=list(csv.DictReader(open('bench/results/latency.csv'))); print(sum(r['flagged']=='True' for r in rows), 'of', len(rows), 'flagged')"`
Expected: zero or very few flagged rows. A high count means the locked clock was set above what the laptop sustains — lower it and re-run.

- [ ] **Step 4: Collect the counter grid**

For each block variant at batch 1 and the largest batch that fits, run `profile_kernel` and `record_counters` into `bench/results/counters.csv`.

- [ ] **Step 5: Write the synthesis**

Create `docs/findings/06-synthesis.md` answering the project's question directly:

1. **Which fusions helped, at which batch sizes, and by how much.** One table.
2. **Which fusions hurt, and which counter explains each.** Cite the specific metric and value.
3. **Where the crossover sits** between launch-bound and bandwidth-bound behaviour, and how it moved the fusion break-even point.
4. **The 1650 Ti versus the Modal GPU**, if the second data point was collected — where the two architectures disagreed and why.
5. **What you would fuse differently** knowing the results.

- [ ] **Step 6: Update the README**

Modify `README.md` to reflect what was actually built: forward-only scope, a link to `docs/findings/06-synthesis.md`, and item (d) restated as future work rather than a current objective.

- [ ] **Step 7: Commit**

```bash
git add bench/run_sweep.py docs/findings/06-synthesis.md README.md bench/results/
git commit -m "feat(bench): add full sweep driver and synthesis findings"
```

---

## Deferred: fp16 experiment

Per the spec, fp16 is a downstream experiment, not part of the main build. After Task 19, if the Task 1 probe showed fp16 `tl.dot` working:

Every kernel already accumulates in fp32 via `.to(tl.float32)` on load, so the fp16 path is a dtype change at the call site plus tightened tolerance review. The measurable question is narrow and worth one findings document: without tensor cores, fp16 halves memory traffic and changes nothing about math throughput, so it should help exactly the bandwidth-bound kernels (LayerNorm, GeLU) and do nothing for the matmuls. Add `dtype` as a sweep axis in `bench/run_sweep.py` and confirm or refute that prediction.

---

## Self-Review

**Spec coverage:**

| Spec section | Task(s) |
|---|---|
| Hardware probe / `tl.dot` risk | 1 |
| Model config, mean pooling, no CLS | 2, 4 |
| Variant registry, enum-equivalent validation | 2 |
| Reference implementation as oracle | 3 |
| Frozen checkpoint | 4 |
| Interleaved harness, clock locking, CSV schema | 5 |
| `ncu` metric set incl. spill counters | 6 |
| Ladder rungs 1–2 (LayerNorm) | 7, 11 |
| Rung 3 (GeLU, negative result) | 8 |
| Rung 4 (Softmax, launch-bound) | 9 |
| Rungs 5–7 (Linear, epilogues) | 10, 12 |
| Rungs 8–10 (attention) | 13, 14, 15 |
| Rungs 12–13 (over-fusion) | 16, 17 |
| End-to-end accuracy + agreement gate | 18 |
| Batch sweep, two-GPU comparison, findings | 19 |
| `nsys` for launch-overhead analysis | 14, 17, 19 |
| fp16 as later experiment | Deferred section |
| Non-power-of-two masking, non-contiguous inputs, numerical stress | 7, 8, 9, 10, 13 |

No spec requirement is unimplemented.

**Type consistency:** `layernorm(x, weight, bias, eps)`, `linear(x, weight, bias)`, `attention(q, k, v, scale)`, `mlp(x, w1, b1, w2, b2)`, and `block(...)` keep identical signatures between `model/baseline/layers.py` and every Triton variant, which is what lets `VariantConfig.resolve` substitute them freely. `layernorm_residual` returns a 2-tuple in both implementations. `BlockParams` attribute names are fixed in Task 4 and consumed unchanged in Tasks 17 and 18.

**Known deliberate item:** Task 4 Step 3 contains `torch.zeros(hidden and dim)`, flagged in the step text as a bug to fix during implementation. It should read `torch.zeros(dim)`.
