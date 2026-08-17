# Hardware Capability: tl.dot on GTX 1650 Ti

## Probe Output

```
GPU:             NVIDIA GeForce GTX 1650 Ti
Compute cap:     7.5
SMs:             16
Total VRAM:      3.90 GB
Shared mem/blk:  49152 B
Torch:           2.11.0+cu128
Triton:          3.6.0

tl.dot torch.float32    OK   max_err=0.00e+00  triton=1.86 TFLOPs  torch=1.79 TFLOPs
tl.dot torch.float16    OK   max_err=0.00e+00  triton=0.65 TFLOPs  torch=0.42 TFLOPs
```

## Task Assignment

Both fp32 and fp16 `tl.dot` operations execute successfully on the GTX 1650 Ti (sm_75). Performance is within ~3× of PyTorch's implementation for both dtypes (fp32 at 1.04× and fp16 at 1.55×), meeting the threshold for local execution. Therefore, all tasks remain on local hardware: tasks 10, 12, 13, 14, 15, 16, 17 (all `tl.dot`-dependent work) run on the 1650 Ti. No work moves to Modal.
