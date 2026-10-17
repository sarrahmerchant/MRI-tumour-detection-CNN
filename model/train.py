from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAINING_DIR = REPO_ROOT / "data" / "Training"

# Normalisation must stay identical to preprocess_image() in prediction_script.py,
# otherwise inference sees a different input distribution than training did.
NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    NORMALIZE,
])

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    NORMALIZE,
])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune ResNet-18 on the MRI training set")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "model.pth",
        help="Where to write the best-validation checkpoint",
    )
    return parser.parse_args()


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def stratified_split(targets: list[int], val_frac: float, seed: int) -> tuple[list[int], list[int]]:
    """Split indices per class, so both sides keep the dataset's class balance."""
    targets_arr = np.asarray(targets)
    rng = np.random.default_rng(seed)
    train_idx: list[int] = []
    val_idx: list[int] = []

    for class_id in np.unique(targets_arr):
        class_idx = rng.permutation(np.flatnonzero(targets_arr == class_id))
        n_val = int(round(val_frac * len(class_idx)))
        val_idx.extend(class_idx[:n_val].tolist())
        train_idx.extend(class_idx[n_val:].tolist())

    return train_idx, val_idx


def run_epoch(model, loader, loss_fn, device, optimizer=None) -> tuple[float, float]:
    """Run one pass. Trains when an optimizer is given, otherwise evaluates."""
    is_train = optimizer is not None
    model.train(is_train)

    running_loss = 0.0
    correct = 0
    seen = 0

    with torch.set_grad_enabled(is_train):
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * labels.size(0)
            correct += int((outputs.argmax(dim=1) == labels).sum().item())
            seen += labels.size(0)

    return running_loss / seen, correct / seen


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = get_device()

    # Two dataset objects over the same folder so the validation half can later be
    # held at EVAL_TRANSFORM even once TRAIN_TRANSFORM gains augmentation.
    train_base = datasets.ImageFolder(root=str(TRAINING_DIR), transform=TRAIN_TRANSFORM)
    val_base = datasets.ImageFolder(root=str(TRAINING_DIR), transform=EVAL_TRANSFORM)
    class_names = [k for k, _ in sorted(train_base.class_to_idx.items(), key=lambda kv: kv[1])]

    train_idx, val_idx = stratified_split(train_base.targets, args.val_frac, args.seed)
    train_loader = DataLoader(
        Subset(train_base, train_idx),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
    )
    val_loader = DataLoader(
        Subset(val_base, val_idx),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"Device: {device}  |  classes: {class_names}")
    print(f"Train: {len(train_idx)} images  |  Val: {len(val_idx)} images")
    print(f"LR: {args.lr}  |  weight decay: {args.weight_decay}  |  epochs: {args.epochs}")
    if args.out.exists():
        print(f"Note: {args.out.name} already exists and will be replaced on improvement.")
    print()

    best_val_acc = -1.0
    best_state = None
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, loss_fn, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, loss_fn, device)
        scheduler.step()

        marker = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            marker = "  <- best"

        print(
            f"Epoch {epoch:>2}/{args.epochs}  "
            f"train loss {train_loss:.4f} acc {100 * train_acc:5.2f}%  |  "
            f"val loss {val_loss:.4f} acc {100 * val_acc:5.2f}%  |  "
            f"lr {optimizer.param_groups[0]['lr']:.2e}{marker}"
        )

    # The last epoch is rarely the best one, so ship the best-validation weights.
    torch.save(best_state, args.out)
    print(f"\nBest val accuracy {100 * best_val_acc:.2f}% at epoch {best_epoch}")
    print(f"Saved weights to: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
