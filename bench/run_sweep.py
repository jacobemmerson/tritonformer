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
from model.baseline.vit import VisionTransformer
from model.config import ViTConfig
from model.registry import Component, VariantConfig, variants
from model.vit import TritonViT

CFG = ViTConfig()
_MODELS: dict[str, TritonViT] = {}
_COMPILED_MODEL: VisionTransformer | None = None


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


def _compiled_model(device: torch.device) -> VisionTransformer:
    """Experiment 2 (docs/findings/08-inductor.md): Inductor's own fusion
    decisions on the plain eager baseline, as an independent check on the
    register rule this project derived from hand-written Triton kernels.

    One compiled module reused across every batch size in the sweep --
    Inductor recompiles per input shape internally (it is shape-specialized
    by default), so this does not need one module per batch; it needs the
    warmup in _forward_compiled to happen before any batch's timed reps,
    which _arms_for_batch's caller (bench/harness.compare()) does not by
    itself guarantee for torch.compile (its 5-call warmup is sized for
    eager/Triton launch overhead, not shape-triggered recompilation, which
    can take seconds to minutes -- far longer than 5 calls' worth of
    tolerance).
    """
    global _COMPILED_MODEL
    if _COMPILED_MODEL is None:
        model = VisionTransformer(CFG).to(device).eval()
        _COMPILED_MODEL = torch.compile(model)
    return _COMPILED_MODEL


@torch.inference_mode()
def _forward_compiled(model: VisionTransformer, images: torch.Tensor,
                      ) -> torch.Tensor:
    return model(images)


def _warm_up_compile(model: VisionTransformer, images: torch.Tensor) -> None:
    """Runs the compiling calls to completion before any timed rep sees
    this batch size's shape, so compile time never lands inside a measured
    latency sample. Inductor recompiles per new input shape; each batch
    size in the sweep pays this once, here, not during compare()."""
    for _ in range(3):
        _forward_compiled(model, images)
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _arms_for_batch(batch: int, dtype: torch.dtype):
    device = torch.device("cuda")
    images = torch.randn(batch, CFG.in_channels, CFG.image_size,
                          CFG.image_size, device=device, dtype=dtype)
    arms = {name: (lambda m=model, im=images: _forward(m, im))
            for name, model in _models(device).items()}

    compiled = _compiled_model(device)
    _warm_up_compile(compiled, images)
    arms["torch_compile"] = lambda m=compiled, im=images: _forward_compiled(m, im)
    return arms


def _bytes_theoretical(batch: int) -> int:
    return batch * CFG.in_channels * CFG.image_size * CFG.image_size * 4


SPEC = RunnerSpec(kernel="vit_forward", arms_for_batch=_arms_for_batch,
                  bytes_theoretical=_bytes_theoretical)


if __name__ == "__main__":
    main(SPEC)
