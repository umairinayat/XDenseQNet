"""
Smoke tests for XDenseQNet models.

Verifies forward-pass shapes, constructor defaults, and basic sanity checks.
Run with:  pytest tests/ -v
"""

import pytest
import torch

# ---------------------------------------------------------------------------
# Skip the entire module gracefully if pennylane/timm are not installed
# (common in CI or lightweight dev environments).
# ---------------------------------------------------------------------------
pennylane = pytest.importorskip("pennylane")
timm = pytest.importorskip("timm")

from models.blocks import (  # noqa: E402
    CBAM,
    ChannelAttention,
    SpatialAttention,
    UniversalCNNExtractor,
    BACKBONE_REGISTRY,
    create_quantum_circuit,
)
from models.proposed import HybridQNN  # noqa: E402
from models.baselines import (  # noqa: E402
    A1_DenseNet_NoQNN,
    A2_FeatureExtractor,
    build_standard_model,
)
from utils.losses import FocalLoss, LabelSmoothingFocalLoss  # noqa: E402
from utils.metrics import compute_metrics  # noqa: E402


DEVICE = torch.device("cpu")
BATCH = 2
NUM_CLASSES = 4
IMG_SHAPE = (BATCH, 3, 224, 224)


# ---------------------------------------------------------------------------
#  Attention blocks
# ---------------------------------------------------------------------------


class TestAttentionBlocks:
    def test_channel_attention_shape(self):
        m = ChannelAttention(128, reduction=16)
        x = torch.randn(BATCH, 128)
        assert m(x).shape == (BATCH, 128)

    def test_spatial_attention_shape(self):
        m = SpatialAttention(256)
        x = torch.randn(BATCH, 256)
        assert m(x).shape == (BATCH, 256)

    def test_cbam_shape(self):
        m = CBAM(512, reduction=16)
        x = torch.randn(BATCH, 512)
        assert m(x).shape == (BATCH, 512)


# ---------------------------------------------------------------------------
#  Backbone extractor (test with lightweight backbones only)
# ---------------------------------------------------------------------------


class TestUniversalCNNExtractor:
    @pytest.mark.parametrize("backbone_key", ["resnet18", "densenet121"])
    def test_extractor_features_only(self, backbone_key):
        ext = UniversalCNNExtractor(backbone_key, num_classes=NUM_CLASSES)
        x = torch.randn(*IMG_SHAPE)
        feats = ext(x, return_features_only=True)
        expected_dim = BACKBONE_REGISTRY[backbone_key]["features"]
        assert feats.shape == (BATCH, expected_dim)

    @pytest.mark.parametrize("backbone_key", ["resnet18"])
    def test_extractor_classification(self, backbone_key):
        ext = UniversalCNNExtractor(backbone_key, num_classes=NUM_CLASSES)
        x = torch.randn(*IMG_SHAPE)
        logits = ext(x, return_features_only=False)
        assert logits.shape == (BATCH, NUM_CLASSES)

    def test_unknown_backbone_raises(self):
        with pytest.raises(ValueError, match="Unknown backbone"):
            UniversalCNNExtractor("fake_backbone")


# ---------------------------------------------------------------------------
#  Quantum circuit
# ---------------------------------------------------------------------------


class TestQuantumCircuit:
    def test_circuit_returns_correct_length(self):
        qnode = create_quantum_circuit(n_qubits=4, n_layers=2)
        inputs = torch.randn(4)
        weights = torch.randn(2, 4, 3)
        result = qnode(inputs, weights)
        assert len(result) == 4

    def test_circuit_values_in_range(self):
        qnode = create_quantum_circuit(n_qubits=3, n_layers=1)
        inputs = torch.randn(3)
        weights = torch.randn(1, 3, 3)
        result = qnode(inputs, weights)
        for val in result:
            assert -1.0 <= float(val) <= 1.0


# ---------------------------------------------------------------------------
#  Proposed model (HybridQNN)
# ---------------------------------------------------------------------------


class TestHybridQNN:
    def test_forward_pretrain_mode(self):
        model = HybridQNN(
            backbone_key="resnet18",
            num_classes=NUM_CLASSES,
            n_qubits=4,
            n_layers=2,
        )
        x = torch.randn(*IMG_SHAPE)
        out = model(x, pretrain_mode=True)
        assert out.shape == (BATCH, NUM_CLASSES)

    def test_forward_full_mode(self):
        model = HybridQNN(
            backbone_key="resnet18",
            num_classes=NUM_CLASSES,
            n_qubits=4,
            n_layers=2,
        )
        x = torch.randn(*IMG_SHAPE)
        out = model(x, pretrain_mode=False)
        assert out.shape == (BATCH, NUM_CLASSES)

    def test_attention_toggle(self):
        model_att = HybridQNN(
            backbone_key="resnet18",
            num_classes=NUM_CLASSES,
            n_qubits=4,
            n_layers=2,
            use_attention=True,
        )
        model_no = HybridQNN(
            backbone_key="resnet18",
            num_classes=NUM_CLASSES,
            n_qubits=4,
            n_layers=2,
            use_attention=False,
        )
        assert model_att.attention is not None
        assert model_no.attention is None


# ---------------------------------------------------------------------------
#  Baseline models
# ---------------------------------------------------------------------------


class TestBaselines:
    def test_a1_forward(self):
        model = A1_DenseNet_NoQNN(num_classes=NUM_CLASSES)
        x = torch.randn(*IMG_SHAPE)
        out = model(x)
        assert out.shape == (BATCH, NUM_CLASSES)

    def test_a1_features_only(self):
        model = A1_DenseNet_NoQNN(num_classes=NUM_CLASSES)
        x = torch.randn(*IMG_SHAPE)
        feats = model(x, return_features_only=True)
        assert feats.shape == (BATCH, 1024)  # DenseNet121

    def test_a2_extractor(self):
        model = A2_FeatureExtractor()
        x = torch.randn(*IMG_SHAPE)
        feats = model(x)
        assert feats.shape == (BATCH, 1024)
        # All params should be frozen
        for p in model.parameters():
            assert not p.requires_grad

    @pytest.mark.parametrize("backbone_key", ["resnet18", "densenet121"])
    def test_build_standard_model(self, backbone_key):
        model = build_standard_model(backbone_key, num_classes=NUM_CLASSES)
        x = torch.randn(*IMG_SHAPE)
        out = model(x)
        assert out.shape == (BATCH, NUM_CLASSES)


# ---------------------------------------------------------------------------
#  Losses
# ---------------------------------------------------------------------------


class TestLosses:
    def test_focal_loss(self):
        loss_fn = FocalLoss(gamma=2.0)
        logits = torch.randn(BATCH, NUM_CLASSES)
        targets = torch.randint(0, NUM_CLASSES, (BATCH,))
        loss = loss_fn(logits, targets)
        assert loss.dim() == 0 and loss.item() >= 0

    def test_label_smoothing_focal_loss(self):
        loss_fn = LabelSmoothingFocalLoss(num_classes=NUM_CLASSES, smoothing=0.1)
        logits = torch.randn(BATCH, NUM_CLASSES)
        targets = torch.randint(0, NUM_CLASSES, (BATCH,))
        loss = loss_fn(logits, targets)
        assert loss.dim() == 0 and loss.item() >= 0


# ---------------------------------------------------------------------------
#  Metrics
# ---------------------------------------------------------------------------


class TestMetrics:
    def test_compute_metrics(self):
        y_true = [0, 1, 2, 3, 0, 1, 2, 3]
        y_pred = [0, 1, 2, 3, 0, 2, 2, 3]
        y_probs = torch.softmax(torch.randn(8, NUM_CLASSES), dim=1).numpy()
        result = compute_metrics(y_true, y_pred, y_probs)
        assert "accuracy" in result
        assert "macro_f1" in result
        assert "roc_auc" in result
        assert "per_class" in result
        assert 0 <= result["accuracy"] <= 1
