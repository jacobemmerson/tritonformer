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
