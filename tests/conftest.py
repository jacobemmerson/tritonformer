import pytest
import torch
from model.config import ViTConfig
from model.baseline import layers  # noqa: F401  (registers "torch" variants)

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
