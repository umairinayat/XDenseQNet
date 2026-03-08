"""
Shared building blocks for XDenseQNet.

Contains:
    - ChannelAttention, SpatialAttention, CBAM (linear-based attention)
    - UniversalCNNExtractor (backbone feature extractor for 8 architectures)
    - create_quantum_circuit (PennyLane variational circuit factory)
"""

import numpy as np
import pennylane as qml
import timm
import torch
import torch.nn as nn
import torchvision.models as models


# ---------------------------------------------------------------------------
#  Attention Modules
# ---------------------------------------------------------------------------


class ChannelAttention(nn.Module):
    """Channel attention via FC squeeze-excitation on 1-D feature vectors."""

    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        reduced = max(in_channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, in_channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.sigmoid(self.fc(x))


class SpatialAttention(nn.Module):
    """Spatial attention via FC bottleneck on 1-D feature vectors."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 4, in_channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(x)


class CBAM(nn.Module):
    """Convolutional Block Attention Module (linear variant for 1-D features).

    Applies channel attention followed by spatial attention.
    """

    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


# ---------------------------------------------------------------------------
#  Backbone registry
# ---------------------------------------------------------------------------

BACKBONE_REGISTRY = {
    "resnet18": {"name": "ResNet18", "features": 512, "type": "torchvision"},
    "resnet50": {"name": "ResNet50", "features": 2048, "type": "torchvision"},
    "densenet121": {"name": "DenseNet121", "features": 1024, "type": "torchvision"},
    "efficientnet_b0": {
        "name": "EfficientNet-B0",
        "features": 1280,
        "type": "torchvision",
    },
    "mobilenet_v2": {"name": "MobileNet-V2", "features": 1280, "type": "torchvision"},
    "vgg16": {"name": "VGG16", "features": 4096, "type": "torchvision"},
    "convnext_tiny": {"name": "ConvNeXt-Tiny", "features": 768, "type": "timm"},
    "swin_tiny": {"name": "Swin-Tiny", "features": 768, "type": "timm"},
}


# ---------------------------------------------------------------------------
#  Universal CNN Feature Extractor
# ---------------------------------------------------------------------------


class UniversalCNNExtractor(nn.Module):
    """Universal feature extractor wrapping torchvision / timm backbones.

    Supports: resnet18, resnet50, densenet121, efficientnet_b0, mobilenet_v2,
              vgg16, convnext_tiny, swin_tiny.

    Parameters
    ----------
    backbone_key : str
        Key into ``BACKBONE_REGISTRY``.
    num_classes : int
        Number of output classes (used only during fine-tuning phase).
    freeze_last_layer : bool
        Whether to freeze the very last trainable layer of the backbone.
    """

    def __init__(
        self,
        backbone_key: str,
        num_classes: int = 4,
        freeze_last_layer: bool = True,
    ):
        super().__init__()

        if backbone_key not in BACKBONE_REGISTRY:
            raise ValueError(
                f"Unknown backbone '{backbone_key}'. "
                f"Choose from: {list(BACKBONE_REGISTRY.keys())}"
            )

        info = BACKBONE_REGISTRY[backbone_key]
        self.backbone_key = backbone_key
        self.backbone_features = info["features"]
        self.backbone_type = info["type"]
        self._freeze_last = freeze_last_layer

        # Load pretrained backbone
        if self.backbone_type == "torchvision":
            self.backbone = self._load_torchvision_backbone()
        else:
            self.backbone = self._load_timm_backbone()

        self._configure_backbone()

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_normalizer = nn.Sequential(
            nn.BatchNorm1d(self.backbone_features),
            nn.ReLU(),
        )
        self.temp_classifier = nn.Linear(self.backbone_features, num_classes)

    # ----- backbone loaders ------------------------------------------------

    def _load_torchvision_backbone(self) -> nn.Module:
        key = self.backbone_key
        if key == "resnet18":
            m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            return nn.Sequential(*list(m.children())[:-1])
        elif key == "resnet50":
            m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            return nn.Sequential(*list(m.children())[:-1])
        elif key == "densenet121":
            m = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
            m.classifier = nn.Identity()
            return m
        elif key == "efficientnet_b0":
            m = models.efficientnet_b0(
                weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
            )
            m.classifier = nn.Identity()
            return m
        elif key == "mobilenet_v2":
            m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
            m.classifier = nn.Identity()
            return m
        elif key == "vgg16":
            m = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
            m.classifier = nn.Sequential(*list(m.classifier.children())[:-1])
            return m
        else:
            raise ValueError(f"Unknown torchvision backbone: {key}")

    def _load_timm_backbone(self) -> nn.Module:
        key = self.backbone_key
        timm_names = {
            "convnext_tiny": "convnext_tiny",
            "swin_tiny": "swin_tiny_patch4_window7_224",
        }
        if key not in timm_names:
            raise ValueError(f"Unknown timm backbone: {key}")
        return timm.create_model(timm_names[key], pretrained=True, num_classes=0)

    # ----- freeze / unfreeze logic -----------------------------------------

    def _configure_backbone(self):
        """Freeze most parameters, unfreeze last block(s) for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = False

        key = self.backbone_key
        if self.backbone_type == "torchvision":
            if "resnet" in key:
                for p in self.backbone[-1].parameters():
                    p.requires_grad = True
            elif "densenet" in key:
                for p in self.backbone.features.denseblock4.parameters():
                    p.requires_grad = True
            elif "efficientnet" in key or "mobilenet" in key:
                total = len(self.backbone.features)
                for i in range(total - 5, total - 1):
                    for p in self.backbone.features[i].parameters():
                        p.requires_grad = True
            elif "vgg" in key:
                for p in list(self.backbone.features.parameters())[-10:]:
                    p.requires_grad = True
        else:  # timm
            if hasattr(self.backbone, "stages"):
                for p in self.backbone.stages[-1].parameters():
                    p.requires_grad = True
            elif hasattr(self.backbone, "blocks"):
                for p in self.backbone.blocks[-2:].parameters():
                    p.requires_grad = True
            elif hasattr(self.backbone, "layers"):
                for p in self.backbone.layers[-1].parameters():
                    p.requires_grad = True

        if self._freeze_last:
            self._freeze_last_layer()

    def _freeze_last_layer(self):
        key = self.backbone_key
        if self.backbone_type == "torchvision":
            if "resnet" in key:
                for p in list(self.backbone[-1].parameters())[-2:]:
                    p.requires_grad = False
            elif "densenet" in key:
                for p in list(self.backbone.features.parameters())[-2:]:
                    p.requires_grad = False
            elif "efficientnet" in key or "mobilenet" in key:
                for p in self.backbone.features[-1].parameters():
                    p.requires_grad = False
        else:
            if hasattr(self.backbone, "stages"):
                for p in list(self.backbone.stages[-1].parameters())[-2:]:
                    p.requires_grad = False

    def unfreeze_for_full_training(self):
        """Unfreeze the entire backbone (for Phase-2 training)."""
        for p in self.backbone.parameters():
            p.requires_grad = True
        if self._freeze_last:
            self._freeze_last_layer()

    # ----- forward ---------------------------------------------------------

    def forward(
        self, x: torch.Tensor, return_features_only: bool = False
    ) -> torch.Tensor:
        features = self.backbone(x)
        if len(features.shape) == 4:
            features = self.adaptive_pool(features)
        features = features.view(features.size(0), -1)
        features = self.feature_normalizer(features)

        if return_features_only:
            return features
        return self.temp_classifier(features)

    def get_output_dim(self) -> int:
        return self.backbone_features


# ---------------------------------------------------------------------------
#  Quantum Circuit Factory
# ---------------------------------------------------------------------------


def create_quantum_circuit(n_qubits: int = 4, n_layers: int = 2):
    """Create a PennyLane variational quantum circuit.

    Architecture:
        - RY angle encoding
        - Parameterised RX / RY / RZ rotation layers
        - Ring CNOT entanglement (+ skip connections on odd layers)
        - PauliZ expectation values

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    n_layers : int
        Number of variational layers.

    Returns
    -------
    qml.QNode
        Differentiable quantum node with ``interface="torch"``.
    """
    dev = qml.device("default.qubit", wires=n_qubits)

    def circuit(inputs, weights):
        # Angle encoding
        for i in range(min(len(inputs), n_qubits)):
            qml.RY(inputs[i] * np.pi, wires=i)

        # Variational layers
        for layer in range(n_layers):
            for i in range(n_qubits):
                qml.RX(weights[layer, i, 0], wires=i)
                qml.RY(weights[layer, i, 1], wires=i)
                qml.RZ(weights[layer, i, 2], wires=i)

            # Ring entanglement
            for i in range(n_qubits):
                qml.CNOT(wires=[i, (i + 1) % n_qubits])

            # Extra skip entanglement on odd layers
            if layer % 2 == 1 and n_qubits > 1:
                for i in range(0, n_qubits - 1, 2):
                    qml.CNOT(wires=[i, i + 1])

        return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

    return qml.QNode(circuit, dev, interface="torch")
