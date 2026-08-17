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
