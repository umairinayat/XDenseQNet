"""
XDenseQNet -- Proposed Hybrid Quantum Neural Network.

DenseNet121 + CBAM Attention + Parameterised Quantum Circuit (4 qubits, 2 layers)
"""

import torch
import torch.nn as nn

from .blocks import CBAM, UniversalCNNExtractor, create_quantum_circuit


class HybridQNN(nn.Module):
    """Hybrid Quantum-Classical Neural Network.

    Pipeline:
        1. UniversalCNNExtractor  ->  classical features  (backbone_dim)
        2. CBAM attention         ->  refined features     (backbone_dim)
        3. Quantum preprocessing  ->  Tanh-compressed      (n_qubits)
        4. PennyLane QNode        ->  PauliZ expectations  (n_qubits)
        5. Concatenation          ->  backbone_dim + n_qubits
        6. Dense classifier       ->  512 -> 256 -> 128 -> 64 -> num_classes

    Args:
        backbone_key:   Key from ``BACKBONE_REGISTRY`` (default ``'densenet121'``).
        num_classes:    Number of output classes.
        n_qubits:       Number of qubits in the quantum circuit.
        n_layers:       Number of variational layers.
        dropout_rate:   Dropout probability in the classifier head.
        use_attention:  Whether to include the CBAM module.
        freeze_last_layer: Freeze the very last backbone sub-layer.
    """

    def __init__(
        self,
        backbone_key: str = "densenet121",
        num_classes: int = 4,
        n_qubits: int = 4,
        n_layers: int = 2,
        dropout_rate: float = 0.5,
        use_attention: bool = True,
        freeze_last_layer: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        # Feature extractor
        self.feature_extractor = UniversalCNNExtractor(
            backbone_key,
            num_classes=num_classes,
            freeze_last_layer=freeze_last_layer,
        )
        self.backbone_output_dim = self.feature_extractor.get_output_dim()

        # Attention
        self.attention = (
            CBAM(self.backbone_output_dim, reduction=16) if use_attention else None
        )

        # Quantum circuit
        self.qnode = create_quantum_circuit(n_qubits, n_layers)
        self.quantum_weights = nn.Parameter(torch.randn(n_layers, n_qubits, 3) * 0.1)
        self.quantum_preprocessing = nn.Sequential(
            nn.Linear(self.backbone_output_dim, n_qubits),
            nn.Tanh(),
        )

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(self.backbone_output_dim + n_qubits, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(64, num_classes),
        )

        self.class_weights = None
        self._initialize_weights()

    # ------------------------------------------------------------------

    def _initialize_weights(self) -> None:
        for m in list(self.classifier.modules()) + list(
            self.quantum_preprocessing.modules()
        ):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def set_class_weights(self, class_weights: torch.Tensor) -> None:
        self.class_weights = class_weights

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, pretrain_mode: bool = False):
        """Forward pass.

        Args:
            x: Input images ``(B, 3, 224, 224)``.
            pretrain_mode: If ``True``, skip quantum+classifier and use the
                temporary classifier inside the backbone extractor (Phase 1).
        """
        if pretrain_mode:
            return self.feature_extractor(x, return_features_only=False)

        # --- Full hybrid pipeline ---
        classical_features = self.feature_extractor(x, return_features_only=True)

        if self.attention is not None:
            classical_features = self.attention(classical_features)

        # Quantum branch
        quantum_input = self.quantum_preprocessing(classical_features)
        quantum_outputs = []
        for i in range(quantum_input.size(0)):
            try:
                q_in = quantum_input[i].detach().cpu()
                q_w = self.quantum_weights.detach().cpu()
                q_out = self.qnode(q_in, q_w)
                quantum_outputs.append(
                    torch.stack([q.clone().detach() for q in q_out]).float()
                )
            except Exception:
                quantum_outputs.append(torch.zeros(self.n_qubits, dtype=torch.float32))

        quantum_features = torch.stack(quantum_outputs).to(x.device)

        # Combine and classify
        combined = torch.cat([classical_features, quantum_features], dim=1)
        return self.classifier(combined)
