import pytest
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model.baseline.vit import VisionTransformer
from model.config import ViTConfig
from model.registry import VariantConfig
from model.vit import TritonViT

CHECKPOINT = "data/checkpoint.pt"
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def load_models(device, variants):
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    cfg = ViTConfig()
    reference = VisionTransformer(cfg).to(device).eval()
    reference.load_state_dict(ckpt["state_dict"])
    triton_model = TritonViT(cfg, variants).to(device).eval()
    triton_model.load_state_dict(ckpt["state_dict"])
    return reference, triton_model, ckpt["test_accuracy"]


def test_logits_match_reference_on_random_input(device):
    variants = VariantConfig(block="triton_composed")
    reference, triton_model, _ = load_models(device, variants)
    images = torch.randn(16, 3, 32, 32, device=device)
    with torch.no_grad():
        torch.testing.assert_close(triton_model(images), reference(images),
                                   rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("block_variant", ["triton_composed", "triton_fused"])
def test_cifar10_accuracy_and_prediction_agreement(device, block_variant):
    """Accuracy alone is too coarse: two models can score identically while
    disagreeing on many images. Both gates must hold."""
    variants = VariantConfig(block=block_variant)
    reference, triton_model, recorded = load_models(device, variants)

    test_tf = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize(MEAN, STD)])
    loader = DataLoader(datasets.CIFAR10("data", train=False, download=True,
                                         transform=test_tf),
                        batch_size=256, shuffle=False)

    correct = agree = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            ref_pred = reference(images).argmax(-1)
            tri_pred = triton_model(images).argmax(-1)
            correct += (tri_pred.cpu() == labels).sum().item()
            agree += (tri_pred == ref_pred).sum().item()
            total += labels.numel()

    assert abs(correct / total - recorded) < 0.001
    assert agree / total >= 0.999
