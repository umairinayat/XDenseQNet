"""
Plotting and visualisation utilities for XDenseQNet.

Functions:
    - plot_confusion_matrix
    - plot_roc_curves
    - plot_per_class_bars
    - save_metrics_csv
"""

import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import label_binarize

CLASS_NAMES = ["Normal", "Monkeypox", "Chickenpox", "Measles"]
NUM_CLASSES = 4

plt.style.use("seaborn-v0_8")


def plot_confusion_matrix(
    cm: np.ndarray,
    title: str,
    save_path: Optional[str] = None,
    class_names: List[str] = None,
    figsize=(10, 8),
) -> None:
    """Plot and optionally save a confusion matrix heatmap."""
    if class_names is None:
        class_names = CLASS_NAMES
    plt.figure(figsize=figsize)
    fmt = "d" if cm.dtype == int else ".1f"
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
        annot_kws={"size": 11},
    )
    plt.title(title, fontsize=14, fontweight="bold")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_roc_curves(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    title: str,
    save_path: Optional[str] = None,
    class_names: List[str] = None,
) -> None:
    """Plot per-class and macro-averaged ROC curves."""
    if class_names is None:
        class_names = CLASS_NAMES
    n_classes = len(class_names)
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))
    colours = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"]

    macro_fpr = np.linspace(0, 1, 200)
    tprs = []

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for i, cls in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
        a = auc(fpr, tpr)
        ax1.plot(
            fpr,
            tpr,
            color=colours[i % len(colours)],
            lw=2,
            label=f"{cls} (AUC={a:.3f})",
        )
        tprs.append(np.interp(macro_fpr, fpr, tpr))

    ax1.plot([0, 1], [0, 1], "k--", lw=1)
    ax1.set(xlabel="FPR", ylabel="TPR", title="Per-Class ROC")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    mean_tpr = np.mean(tprs, axis=0)
    ma = auc(macro_fpr, mean_tpr)
    ax2.plot(macro_fpr, mean_tpr, "navy", lw=2.5, label=f"Macro AUC = {ma:.4f}")
    ax2.fill_between(macro_fpr, mean_tpr, alpha=0.1, color="navy")
    ax2.plot([0, 1], [0, 1], "k--", lw=1)
    ax2.set(xlabel="FPR", ylabel="TPR", title="Macro-avg ROC")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_per_class_bars(
    per_class: Dict[str, Dict[str, float]],
    title: str,
    save_path: Optional[str] = None,
    class_names: List[str] = None,
) -> None:
    """Bar chart of per-class precision, recall, F1, and specificity."""
    if class_names is None:
        class_names = CLASS_NAMES
    metrics = ["precision", "sensitivity", "f1_score", "specificity"]
    labels = ["Precision", "Recall", "F1", "Specificity"]
    colours = ["#3498db", "#2ecc71", "#e74c3c", "#9b59b6"]

    x = np.arange(len(class_names))
    w = 0.2

    fig, ax = plt.subplots(figsize=(11, 5))
    for j, (m, lab, col) in enumerate(zip(metrics, labels, colours)):
        vals = [per_class[c][m] for c in class_names]
        bars = ax.bar(x + (j - 1.5) * w, vals, w, label=lab, color=col, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(class_names, fontsize=10)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Score")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_metrics_csv(
    metrics: Dict,
    model_name: str,
    inference_time_ms: float,
    out_dir: str,
    suffix: str = "",
    class_names: List[str] = None,
) -> str:
    """Save metrics to a CSV file. Returns the output path."""
    if class_names is None:
        class_names = CLASS_NAMES
    rows = [
        {"Metric": "Model", "Value": model_name},
        {"Metric": "Accuracy", "Value": f"{metrics['accuracy']:.4f}"},
        {"Metric": "Macro F1", "Value": f"{metrics['macro_f1']:.4f}"},
        {"Metric": "Precision", "Value": f"{metrics['macro_precision']:.4f}"},
        {"Metric": "Recall", "Value": f"{metrics['macro_recall']:.4f}"},
        {"Metric": "ROC-AUC", "Value": f"{metrics['roc_auc']:.4f}"},
        {"Metric": "Infer ms/img", "Value": f"{inference_time_ms:.3f}"},
    ]
    per_class = metrics.get("per_class", {})
    for cls in class_names:
        if cls in per_class:
            m = per_class[cls]
            rows.extend(
                [
                    {"Metric": f"{cls} F1", "Value": f"{m['f1_score']:.4f}"},
                    {"Metric": f"{cls} Prec", "Value": f"{m['precision']:.4f}"},
                    {"Metric": f"{cls} Rec", "Value": f"{m['sensitivity']:.4f}"},
                    {"Metric": f"{cls} Spec", "Value": f"{m['specificity']:.4f}"},
                ]
            )

    df = pd.DataFrame(rows)
    path = os.path.join(out_dir, f"metrics{suffix}.csv")
    df.to_csv(path, index=False)
    return path
