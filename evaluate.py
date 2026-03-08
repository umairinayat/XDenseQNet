#!/usr/bin/env python3
"""
XDenseQNet -- Evaluation Script

Loads a trained checkpoint and evaluates on the test set.
Generates confusion matrix, ROC curves, per-class bar charts, and CSV metrics.

Usage:
    python evaluate.py --checkpoint checkpoints/best.pth --config configs/proposed.yaml
    python evaluate.py --checkpoint checkpoints/best.pth --config configs/proposed.yaml --save-dir results/
"""

import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import yaml

from models import HybridQNN
from utils.augmentation import create_dataloaders
from utils.metrics import compute_metrics, CLASS_NAMES
from utils.visualization import (
    plot_confusion_matrix,
    plot_roc_curves,
    plot_per_class_bars,
    save_metrics_csv,
)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate_model(model, loader, device):
    """Run full evaluation, returning metrics + timing info."""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    total_time = 0.0
    n_images = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        batch_size = images.size(0)

        t0 = time.time()
        outputs = model(images, pretrain_mode=False)
        total_time += time.time() - t0

        probs = torch.softmax(outputs, dim=1)
        _, preds = outputs.max(1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        n_images += batch_size

    metrics = compute_metrics(
        np.array(all_labels), np.array(all_preds), np.array(all_probs)
    )
    inference_ms = (total_time / n_images) * 1000 if n_images > 0 else 0.0
    return metrics, np.array(all_labels), np.array(all_probs), inference_ms


def main():
    parser = argparse.ArgumentParser(description="XDenseQNet Evaluation")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pth)",
    )
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
        help="Device (e.g. cuda, cuda:0, cpu)",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="results",
        help="Directory to save evaluation outputs",
    )
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 42))

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data loaders
    data_cfg = cfg["data"]
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

    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    print(f"  Checkpoint epoch: {ckpt.get('epoch', '?')}")
    print(f"  Checkpoint val_acc: {ckpt.get('val_accuracy', '?')}")

    # Evaluate
    print("\nEvaluating on test set...")
    metrics, y_true, y_probs, inference_ms = evaluate_model(
        model, loaders["test_loader"], device
    )

    # Print results
    backbone = model_cfg["backbone"]
    model_name = (
        f"{backbone} + "
        f"{'CBAM + ' if model_cfg.get('use_attention') else ''}"
        f"{model_cfg['n_qubits']}Q-{model_cfg['n_layers']}L"
    )
    print(f"\n{'=' * 60}")
    print(f"  Model:     {model_name}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy'] * 100:.2f}%)")
    print(f"  Macro F1:  {metrics['macro_f1']:.4f}")
    print(f"  Precision: {metrics['macro_precision']:.4f}")
    print(f"  Recall:    {metrics['macro_recall']:.4f}")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"  Inference: {inference_ms:.2f} ms/image")
    print(f"{'=' * 60}")

    print("\n  Per-class results:")
    for cls, m in metrics["per_class"].items():
        print(
            f"    {cls:12s}  F1={m['f1_score']:.4f}  "
            f"Prec={m['precision']:.4f}  Rec={m['sensitivity']:.4f}  "
            f"Spec={m['specificity']:.4f}"
        )

    # Save outputs
    fig_dir = os.path.join(args.save_dir, "figures")
    tbl_dir = os.path.join(args.save_dir, "tables")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(tbl_dir, exist_ok=True)

    safe_name = backbone.replace("/", "_")

    # Confusion matrix
    cm_path = os.path.join(fig_dir, f"confusion_matrix_{safe_name}.png")
    plot_confusion_matrix(metrics["confusion_matrix"], model_name, save_path=cm_path)
    print(f"\n  Saved: {cm_path}")

    # ROC curves
    y_pred = np.argmax(y_probs, axis=1)
    roc_path = os.path.join(fig_dir, f"roc_curves_{safe_name}.png")
    plot_roc_curves(y_true, y_probs, model_name, save_path=roc_path)
    print(f"  Saved: {roc_path}")

    # Per-class bar chart
    bar_path = os.path.join(fig_dir, f"per_class_{safe_name}.png")
    plot_per_class_bars(metrics["per_class"], model_name, save_path=bar_path)
    print(f"  Saved: {bar_path}")

    # CSV
    csv_path = save_metrics_csv(
        metrics, model_name, inference_ms, tbl_dir, suffix=f"_{safe_name}"
    )
    print(f"  Saved: {csv_path}")

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
