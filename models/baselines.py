"""
Baseline and ablation models for XDenseQNet.

A1 -- DenseNet121 + CBAM + Dense Head + Softmax  (no quantum branch)
A2 -- DenseNet121 + CBAM feature extractor  ->  classical ML classifiers
A3 -- Standard backbone + single linear head  (no CBAM, no quantum)
"""

import timm
import torch
import torch.nn as nn
import torchvision.models as models

from .blocks import CBAM, UniversalCNNExtractor, BACKBONE_REGISTRY


# ---------------------------------------------------------------------------
#  A1: DenseNet121 + CBAM + Deep Dense Head (no QNN)
# ---------------------------------------------------------------------------


class A1_DenseNet_NoQNN(nn.Module):
    """Ablation A1 -- identical to HybridQNN but without the quantum branch.

    Architecture:
        UniversalCNNExtractor (DenseNet121) -> CBAM -> 512->256->128->64->4
    """

    def __init__(self, num_classes: int = 4, dropout: float = 0.5):
        super().__init__()
        self.feature_extractor = UniversalCNNExtractor(
            "densenet121", num_classes=num_classes
        )
        feat_dim = self.feature_extractor.get_output_dim()  # 1024

        self.attention = CBAM(feat_dim, reduction=16)

        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, num_classes),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, return_features_only: bool = False):
        f = self.feature_extractor(x, return_features_only=True)
        f = self.attention(f)
        if return_features_only:
            return f
        return self.classifier(f)


# ---------------------------------------------------------------------------
#  A2: Frozen DenseNet121 + CBAM -> Feature extractor for classical ML
# ---------------------------------------------------------------------------


class A2_FeatureExtractor(nn.Module):
    """Frozen DenseNet121 + CBAM used only to extract 1024-d feature vectors.

    All parameters are frozen (ImageNet weights only, no fine-tuning).
    Use with sklearn classifiers: SVM, RandomForest, KNN, etc.
    """

    def __init__(self):
        super().__init__()
        self.feature_extractor = UniversalCNNExtractor("densenet121")
        self.attention = CBAM(self.feature_extractor.get_output_dim(), reduction=16)
        # Freeze everything
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.feature_extractor(x, return_features_only=True)
        return self.attention(f)  # (B, 1024)


# ---------------------------------------------------------------------------
#  A3: Standard backbone + linear head  (no CBAM, no QNN)
# ---------------------------------------------------------------------------


def build_standard_model(backbone_key: str, num_classes: int = 4) -> nn.Module:
    """Build a standard torchvision/timm model with a simple linear head.

    This is used for Ablation A3 -- comparing against the full XDenseQNet pipeline.
    """
    if backbone_key not in BACKBONE_REGISTRY:
        raise ValueError(f"Unknown backbone '{backbone_key}'.")

    if backbone_key == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = nn.Linear(512, num_classes)
    elif backbone_key == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        m.fc = nn.Linear(2048, num_classes)
    elif backbone_key == "densenet121":
        m = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        m.classifier = nn.Linear(1024, num_classes)
    elif backbone_key == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        m.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(1280, num_classes))
    elif backbone_key == "mobilenet_v2":
        m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        m.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(1280, num_classes))
    elif backbone_key == "vgg16":
        m = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        m.classifier[-1] = nn.Linear(4096, num_classes)
    elif backbone_key == "convnext_tiny":
        m = timm.create_model("convnext_tiny", pretrained=True, num_classes=num_classes)
    elif backbone_key == "swin_tiny":
        m = timm.create_model(
            "swin_tiny_patch4_window7_224", pretrained=True, num_classes=num_classes
        )
    else:
        raise ValueError(f"No A3 builder for backbone '{backbone_key}'.")
    return m


def get_a3_param_groups(model: nn.Module, lr: float):
    """Separate backbone vs head parameters for differential learning rates."""
    head_params, backbone_params = [], []
    for name, p in model.named_parameters():
        if any(k in name for k in ("fc", "classifier", "head", "norm")):
            head_params.append(p)
        else:
            backbone_params.append(p)
    return [
        {"params": backbone_params, "lr": lr * 0.01, "name": "backbone"},
        {"params": head_params, "lr": lr, "name": "head"},
    ]
