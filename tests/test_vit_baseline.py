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
