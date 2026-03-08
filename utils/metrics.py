"""
Evaluation metrics for XDenseQNet.

Provides ``compute_metrics`` for per-class and macro-averaged metrics.
"""

from typing import Dict, Any

import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize

CLASS_NAMES = ["Normal", "Monkeypox", "Chickenpox", "Measles"]
NUM_CLASSES = 4


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
    class_names: list = None,
    num_classes: int = None,
) -> Dict[str, Any]:
    """Compute comprehensive classification metrics.

    Args:
        y_true:  Ground-truth labels ``(N,)``.
        y_pred:  Predicted labels ``(N,)``.
        y_probs: Predicted probabilities ``(N, C)``.
        class_names: List of class name strings.
        num_classes:  Number of classes.

    Returns:
        Dictionary with accuracy, macro F1/precision/recall, ROC-AUC,
        per-class metrics, and confusion matrix.
    """
    if class_names is None:
        class_names = CLASS_NAMES
    if num_classes is None:
        num_classes = NUM_CLASSES

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    per_class = {}
    for i, cls in enumerate(class_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = (
            2 * precision * sensitivity / (precision + sensitivity)
            if (precision + sensitivity) > 0
            else 0.0
        )

        per_class[cls] = {
            "sensitivity": sensitivity,
            "specificity": specificity,
            "precision": precision,
            "f1_score": f1,
            "TP": int(tp),
            "FP": int(fp),
            "FN": int(fn),
            "TN": int(tn),
        }

    # ROC-AUC (macro, one-vs-rest)
    try:
        y_bin = label_binarize(y_true, classes=list(range(num_classes)))
        roc_auc = float(
            roc_auc_score(y_bin, y_probs, average="macro", multi_class="ovr")
        )
    except Exception:
        roc_auc = float("nan")

    return {
        "accuracy": float(np.mean(np.array(y_pred) == np.array(y_true))),
        "macro_f1": float(np.mean([m["f1_score"] for m in per_class.values()])),
        "macro_precision": float(np.mean([m["precision"] for m in per_class.values()])),
        "macro_recall": float(np.mean([m["sensitivity"] for m in per_class.values()])),
        "roc_auc": roc_auc,
        "per_class": per_class,
        "confusion_matrix": cm,
    }
