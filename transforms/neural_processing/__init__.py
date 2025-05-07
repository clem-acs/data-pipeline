"""
Neural processing and windowing module for EEG and fNIRS data.

This package provides functions and classes for preprocessing, windowing,
and post-processing EEG and fNIRS data. It includes PyTorch integration
for machine learning pipelines.
"""

# Import main components
from .eeg_preprocessing import preprocess_eeg
from .fnirs_preprocessing import preprocess_fnirs, get_module_channel_indices
from .windowing import create_windows
from .postprocessing import postprocess_windows
from .window_dataset import WindowDataset

__all__ = [
    'preprocess_eeg',
    'preprocess_fnirs',
    'get_module_channel_indices',
    'create_windows',
    'postprocess_windows',
    'WindowDataset'
]