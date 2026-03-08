#!/usr/bin/env python3
"""
XDenseQNet -- Training Script

Two-phase training pipeline:
  Phase 1: Backbone fine-tuning with CrossEntropy + ReduceLROnPlateau
  Phase 2: Full hybrid training with LabelSmoothingFocalLoss + CosineAnnealing + Mixup

Usage:
    python train.py --config configs/proposed.yaml
    python train.py --config configs/baselines/resnet50.yaml --device cuda:1
"""

import argparse
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from models import HybridQNN
from utils.augmentation import create_dataloaders, organize_balanced_dataset
from utils.losses import FocalLoss, LabelSmoothingFocalLoss
from utils.metrics import compute_metrics


# ---------------------------------------------------------------------------
#  Reproducibility
# ---------------------------------------------------------------------------


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# ---------------------------------------------------------------------------
#  Mixup augmentation
# ---------------------------------------------------------------------------


def mixup_data(x, y, alpha=0.2):
    """Apply Mixup augmentation to a batch."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute Mixup loss."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ---------------------------------------------------------------------------
#  Training loop helpers
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    pretrain_mode: bool = False,
    use_mixup: bool = False,
    mixup_alpha: float = 0.2,
    grad_clip: float = 0.0,
) -> dict:
    """Train for one epoch. Returns dict with loss, accuracy."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        if use_mixup and not pretrain_mode:
            images, y_a, y_b, lam = mixup_data(images, labels, mixup_alpha)
            outputs = model(images, pretrain_mode=pretrain_mode)
            loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
        else:
            outputs = model(images, pretrain_mode=pretrain_mode)
            loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    return {
        "loss": running_loss / total,
        "accuracy": correct / total,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    pretrain_mode: bool = False,
) -> dict:
    """Validate model. Returns dict with loss, accuracy, predictions."""
    model.eval()
    running_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images, pretrain_mode=pretrain_mode)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        probs = torch.softmax(outputs, dim=1)
        _, preds = outputs.max(1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    total = len(all_labels)
    metrics = compute_metrics(
        np.array(all_labels), np.array(all_preds), np.array(all_probs)
    )
    metrics["loss"] = running_loss / total
    return metrics


# ---------------------------------------------------------------------------
#  Phase 1: Backbone fine-tuning
# ---------------------------------------------------------------------------


def phase1_train(model, loaders, cfg, device):
    """Phase 1: Fine-tune backbone with frozen quantum layers."""
    p1 = cfg["phase1"]
    print("\n" + "=" * 70)
    print("PHASE 1: Backbone Fine-Tuning")
    print(f"  Epochs: {p1['epochs']}, LR: {p1['learning_rate']}")
    print("=" * 70)

    criterion = nn.CrossEntropyLoss(
        weight=loaders["train_dataset"].get_class_weights().to(device)
    )

    # Only optimise backbone + temp_classifier parameters
    params = [
        {"params": model.feature_extractor.parameters(), "lr": p1["learning_rate"]},
        {
            "params": model.feature_extractor.temp_classifier.parameters(),
            "lr": p1["learning_rate"],
        },
    ]
    optimizer = torch.optim.Adam(
        params, lr=p1["learning_rate"], weight_decay=p1.get("weight_decay", 1e-3)
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=p1.get("scheduler_factor", 0.5),
        patience=p1.get("scheduler_patience", 3),
    )

    best_val_loss = float("inf")
    for epoch in range(1, p1["epochs"] + 1):
        t0 = time.time()
        train_stats = train_one_epoch(
            model,
            loaders["train_loader"],
            criterion,
            optimizer,
            device,
            pretrain_mode=True,
        )
        val_stats = validate(
            model, loaders["val_loader"], criterion, device, pretrain_mode=True
        )
        scheduler.step(val_stats["loss"])

        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"  [{epoch:02d}/{p1['epochs']}] "
            f"train_loss={train_stats['loss']:.4f} "
            f"train_acc={train_stats['accuracy']:.4f} | "
            f"val_loss={val_stats['loss']:.4f} "
            f"val_acc={val_stats['accuracy']:.4f} "
            f"lr={lr_now:.1e} ({elapsed:.1f}s)"
        )

        if val_stats["loss"] < best_val_loss:
            best_val_loss = val_stats["loss"]

    return model


# ---------------------------------------------------------------------------
#  Phase 2: Full hybrid training
# ---------------------------------------------------------------------------


def phase2_train(model, loaders, cfg, device, checkpoint_dir):
    """Phase 2: Full hybrid training with quantum circuit."""
    p2 = cfg["phase2"]
    print("\n" + "=" * 70)
    print("PHASE 2: Full Hybrid Training")
    print(f"  Epochs: {p2['epochs']}, LR: {p2['learning_rate']}")
    print(f"  Unfreeze at epoch: {p2.get('unfreeze_epoch', 15)}")
    print("=" * 70)

    class_weights = loaders["train_dataset"].get_class_weights().to(device)
    criterion = LabelSmoothingFocalLoss(
        num_classes=cfg["model"]["num_classes"],
        alpha=class_weights,
        gamma=p2.get("focal_gamma", 2.0),
        smoothing=p2.get("label_smoothing", 0.1),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=p2["learning_rate"],
        weight_decay=p2.get("weight_decay", 1e-3),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=p2.get("cosine_T_0", 10),
        T_mult=p2.get("cosine_T_mult", 2),
        eta_min=p2.get("cosine_eta_min", 1e-7),
    )

    use_mixup = p2.get("use_mixup", True)
    mixup_alpha = p2.get("mixup_alpha", 0.2)
    grad_clip = p2.get("grad_clip_max_norm", 1.0)
    unfreeze_epoch = p2.get("unfreeze_epoch", 15)
    patience = p2.get("patience", 20)

    best_val_acc = 0.0
    no_improve = 0

    for epoch in range(1, p2["epochs"] + 1):
        # Unfreeze backbone at specified epoch
        if epoch == unfreeze_epoch:
            print(f"\n  >> Unfreezing backbone at epoch {epoch}")
            model.feature_extractor.unfreeze_for_full_training()
            # Lower backbone LR
            mult = p2.get("unfreeze_lr_multiplier", 0.01)
            for pg in optimizer.param_groups:
                pg["lr"] = p2["learning_rate"] * mult

        t0 = time.time()
        train_stats = train_one_epoch(
            model,
            loaders["train_loader"],
            criterion,
            optimizer,
            device,
            pretrain_mode=False,
            use_mixup=use_mixup,
            mixup_alpha=mixup_alpha,
            grad_clip=grad_clip,
        )
        val_stats = validate(
            model, loaders["val_loader"], criterion, device, pretrain_mode=False
        )
        scheduler.step()

        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"  [{epoch:02d}/{p2['epochs']}] "
            f"train_loss={train_stats['loss']:.4f} "
            f"train_acc={train_stats['accuracy']:.4f} | "
            f"val_loss={val_stats['loss']:.4f} "
            f"val_acc={val_stats['accuracy']:.4f} "
            f"val_f1={val_stats['macro_f1']:.4f} "
            f"lr={lr_now:.1e} ({elapsed:.1f}s)"
        )

        # Save best checkpoint
        if val_stats["accuracy"] > best_val_acc:
            best_val_acc = val_stats["accuracy"]
            no_improve = 0
            ckpt_path = os.path.join(checkpoint_dir, "best.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_accuracy": best_val_acc,
                    "val_f1": val_stats["macro_f1"],
                    "config": cfg,
                },
                ckpt_path,
            )
            print(f"    >> Saved best model (acc={best_val_acc:.4f})")
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"\n  Early stopping at epoch {epoch} (patience={patience})")
            break

    return model, best_val_acc


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="XDenseQNet Training")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/proposed.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (e.g. cuda, cuda:0, cpu). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--skip-phase1",
        action="store_true",
        help="Skip Phase 1 backbone fine-tuning",
    )
    parser.add_argument(
        "--prepare-data",
        action="store_true",
        help="Run balanced data augmentation before training",
    )
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    seed = cfg.get("seed", 42)
    set_seed(seed)

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Data preparation (optional)
    data_cfg = cfg["data"]
    if args.prepare_data:
        print("\nPreparing balanced dataset...")
        organize_balanced_dataset(
            source_path=data_cfg["dataset_path"],
            target_path=data_cfg["processed_path"],
            target_samples=data_cfg.get("target_samples_per_class", 550),
            train_ratio=data_cfg.get("train_ratio", 0.70),
            val_ratio=data_cfg.get("val_ratio", 0.15),
            image_size=tuple(data_cfg.get("image_size", [224, 224])),
        )

    # Data loaders
    print("\nLoading datasets...")
    loaders = create_dataloaders(
        processed_dir=data_cfg["processed_path"],
        batch_size=data_cfg.get("batch_size", 16),
        image_size=tuple(data_cfg.get("image_size", [224, 224])),
        num_workers=data_cfg.get("num_workers", 0),
    )

    # Build model
    model_cfg = cfg["model"]
    model = HybridQNN(
        backbone_key=model_cfg["backbone"],
        num_classes=model_cfg["num_classes"],
        n_qubits=model_cfg["n_qubits"],
        n_layers=model_cfg["n_layers"],
        dropout_rate=model_cfg.get("dropout_rate", 0.5),
        use_attention=model_cfg.get("use_attention", True),
        freeze_last_layer=model_cfg.get("freeze_last_layer", True),
    )
    model = model.to(device)

    class_weights = loaders["train_dataset"].get_class_weights().to(device)
    model.set_class_weights(class_weights)

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"\nModel: {model_cfg['backbone']} + "
        f"{'CBAM + ' if model_cfg.get('use_attention') else ''}"
        f"{model_cfg['n_qubits']}Q-{model_cfg['n_layers']}L QNN"
    )
    print(f"Parameters: {total_params:,} total, {trainable:,} trainable")

    # Phase 1
    if not args.skip_phase1:
        model = phase1_train(model, loaders, cfg, device)

    # Phase 2
    model, best_acc = phase2_train(model, loaders, cfg, device, args.checkpoint_dir)

    # Final evaluation on test set
    print("\n" + "=" * 70)
    print("FINAL EVALUATION ON TEST SET")
    print("=" * 70)

    # Load best checkpoint
    best_ckpt = os.path.join(args.checkpoint_dir, "best.pth")
    if os.path.exists(best_ckpt):
        ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"  Loaded best checkpoint (epoch {ckpt['epoch']})")

    criterion = nn.CrossEntropyLoss()
    test_metrics = validate(model, loaders["test_loader"], criterion, device)

    print(f"\n  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  Macro F1:  {test_metrics['macro_f1']:.4f}")
    print(f"  Precision: {test_metrics['macro_precision']:.4f}")
    print(f"  Recall:    {test_metrics['macro_recall']:.4f}")
    print(f"  ROC-AUC:   {test_metrics['roc_auc']:.4f}")

    # Per-class breakdown
    print("\n  Per-class results:")
    for cls, m in test_metrics["per_class"].items():
        print(
            f"    {cls:12s}  F1={m['f1_score']:.4f}  "
            f"Prec={m['precision']:.4f}  Rec={m['sensitivity']:.4f}  "
            f"Spec={m['specificity']:.4f}"
        )

    print(f"\nTraining complete. Best checkpoint: {best_ckpt}")


if __name__ == "__main__":
    main()
