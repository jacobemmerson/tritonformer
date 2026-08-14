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
