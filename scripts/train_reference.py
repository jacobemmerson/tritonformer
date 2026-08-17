"""Trains the reference ViT and freezes the checkpoint.

Run this ONCE. Every accuracy comparison for the life of the project
comes from the resulting file; retraining invalidates historical numbers.
"""
import argparse
import sys
from pathlib import Path

# pyproject.toml's pythonpath applies only to pytest, not direct execution;
# add the repo root so `model.*` resolves when this script is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model.baseline.vit import VisionTransformer
from model.config import ViTConfig

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def loaders(root, batch_size, num_workers):
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    train = datasets.CIFAR10(root, train=True, download=True, transform=train_tf)
    test = datasets.CIFAR10(root, train=False, download=True, transform=test_tf)
    return (DataLoader(train, batch_size, shuffle=True, num_workers=num_workers,
                       drop_last=True),
            DataLoader(test, 512, shuffle=False, num_workers=num_workers))


@torch.no_grad()
def accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    for images, labels in loader:
        preds = model(images.to(device)).argmax(-1).cpu()
        correct += (preds == labels).sum().item()
        total += labels.numel()
    return correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--out", default="data/checkpoint.pt")
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda")
    cfg = ViTConfig()
    model = VisionTransformer(cfg).to(device)
    train_loader, test_loader = loaders(args.data_root, args.batch_size,
                                         args.num_workers)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, epochs=args.epochs,
        steps_per_epoch=len(train_loader))

    for epoch in range(args.epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            loss = F.cross_entropy(model(images), labels, label_smoothing=0.1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        if (epoch + 1) % 10 == 0:
            print(f"epoch {epoch + 1}: test acc {accuracy(model, test_loader, device):.4f}")

    final = accuracy(model, test_loader, device)
    torch.save({"state_dict": model.state_dict(), "cfg": cfg,
                "test_accuracy": final}, args.out)
    print(f"saved {args.out} with test accuracy {final:.4f}")


if __name__ == "__main__":
    main()
