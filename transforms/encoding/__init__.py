"""
Efficient neural encoding module for data pipeline.

This module provides optimized implementations for neural encoding and classification:
- Efficient data loading with label pre-processing
- Optimized model architectures
- Training utilities
"""

from .dataloader import (
    EfficientQueryDataset,
    create_efficient_data_loaders,
    filter_binary_labels,
    filter_labels_in_range,
)

from .models import (
    EfficientEEGClassifier,
    DualModalityModel,
    SimpleEncoder,
)

__all__ = [
    'EfficientQueryDataset',
    'create_efficient_data_loaders',
    'filter_binary_labels',
    'filter_labels_in_range',
    'EfficientEEGClassifier',
    'DualModalityModel',
    'SimpleEncoder',
]