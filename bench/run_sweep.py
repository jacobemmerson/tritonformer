"""Full measurement grid: the end-to-end ViT forward pass across every
registered block variant and the batch sweep.

Complements the per-kernel benchmarks (bench/run_layernorm.py,
bench/run_block.py, ...), which isolate one op or one block call each.
This sweeps model.forward(), so the launch-count and DRAM round-trip
differences between block variants show up as observed end-to-end
latency rather than being inferred from the per-kernel numbers alone.

Reuses bench/runner.py's sweep/single-shot machinery (OOM truncation,
CSV recording, and the --kernel/--variant/--batch/--dtype single-shot
contract that bench/profile.py::profile_kernel drives ncu with) so this
grid gets the same reproducibility guarantees as every other kernel here.

Truncates the batch sweep on torch.cuda.OutOfMemoryError rather than
failing, recording the ceiling reached -- 4GB may not fit every block
variant at batch 512.
"""
import torch

from bench.runner import RunnerSpec, main
from model.config import ViTConfig
from model.registry import Component, VariantConfig, variants
from model.vit import TritonViT

CFG = ViTConfig()
_MODELS: dict[str, TritonViT] = {}


def _models(device: torch.device) -> dict[str, TritonViT]:
    if not _MODELS:
        for variant in variants(Component.BLOCK):
            try:
                _MODELS[variant] = TritonViT(
                    CFG, VariantConfig(block=variant)).to(device).eval()
            except ValueError:
                continue
    return _MODELS


@torch.inference_mode()
def _forward(model: TritonViT, images: torch.Tensor) -> torch.Tensor:
    return model(images)


def _arms_for_batch(batch: int, dtype: torch.dtype):
    device = torch.device("cuda")
    images = torch.randn(batch, CFG.in_channels, CFG.image_size,
                          CFG.image_size, device=device, dtype=dtype)
    return {name: (lambda m=model, im=images: _forward(m, im))
            for name, model in _models(device).items()}


def _bytes_theoretical(batch: int) -> int:
    return batch * CFG.in_channels * CFG.image_size * CFG.image_size * 4


SPEC = RunnerSpec(kernel="vit_forward", arms_for_batch=_arms_for_batch,
                  bytes_theoretical=_bytes_theoretical)


if __name__ == "__main__":
    main(SPEC)
