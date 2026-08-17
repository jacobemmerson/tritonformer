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
        self.b2 = nn.Parameter(torch.zeros(dim))
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
