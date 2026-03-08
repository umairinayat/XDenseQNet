"""XDenseQNet model package."""

from .blocks import (
    BACKBONE_REGISTRY,
    CBAM,
    ChannelAttention,
    SpatialAttention,
    UniversalCNNExtractor,
    create_quantum_circuit,
)
from .proposed import HybridQNN
from .baselines import (
    A1_DenseNet_NoQNN,
    A2_FeatureExtractor,
    build_standard_model,
    get_a3_param_groups,
)

__all__ = [
    "BACKBONE_REGISTRY",
    "CBAM",
    "ChannelAttention",
    "SpatialAttention",
    "UniversalCNNExtractor",
    "create_quantum_circuit",
    "HybridQNN",
    "A1_DenseNet_NoQNN",
    "A2_FeatureExtractor",
    "build_standard_model",
    "get_a3_param_groups",
]
