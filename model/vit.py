import torch
from torch import Tensor, nn

import model.baseline  # noqa: F401  (registers "torch" variants)
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
