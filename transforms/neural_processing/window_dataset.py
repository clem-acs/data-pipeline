"""
WindowDataset class for storing and accessing windowed neural data.

This implements a PyTorch-compatible Dataset for neural data windows,
with fixed window size matching physiological frame boundaries.
The dataset handles timestamp alignment and gaps in fNIRS data.
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Union, Tuple

# Try to import torch, but don't fail if it's not available
try:
    import torch
    from torch.utils.data import Dataset
except ImportError:
    torch = None
    # Define a minimal Dataset base class if torch is not available
    class Dataset:
        def __len__(self):
            return 0
        def __getitem__(self, idx):
            raise NotImplementedError

# Import the windowing function
try:
    # Try relative import first
    from .windowing import create_windows
except ImportError:
    # Fall back to direct import if relative import fails
    from windowing import create_windows

# Set up logger
logger = logging.getLogger(__name__)


class WindowDataset(Dataset):
    """
    PyTorch Dataset for windowed neural data.

    This represents a collection of time-aligned windows containing
    EEG and fNIRS data that can be used directly with PyTorch DataLoader.
    Each window maintains its original sampling rate and frame alignment.

    The dataset creates windows that properly align EEG and fNIRS data
    by timestamp and handles cases where fNIRS frames are skipped.
    """

    def __init__(self, eeg_data: Optional[np.ndarray] = None, 
                 fnirs_data: Optional[np.ndarray] = None, 
                 eeg_timestamps: Optional[np.ndarray] = None, 
                 fnirs_timestamps: Optional[np.ndarray] = None,
                 precompute_windows: bool = False):
        """
        Initialize a WindowDataset with original data.

        Args:
            eeg_data: EEG data array with shape (frames, channels, samples_per_frame)
            fnirs_data: fNIRS data array with shape (frames, channels, 1)
            eeg_timestamps: Timestamps for EEG data
            fnirs_timestamps: Timestamps for fNIRS data
            precompute_windows: Whether to precompute all windows at initialization
                               (default: False = lazy loading)
        """
        # Store the original data
        self.eeg_data = eeg_data
        self.fnirs_data = fnirs_data
        self.eeg_timestamps = eeg_timestamps
        self.fnirs_timestamps = fnirs_timestamps

        # Fixed parameters for the windowing
        self.eeg_frames_per_window = 7  # 7 EEG frames per window
        self.fnirs_frames_per_window = 1  # 1 fNIRS frame per window
        self.window_size_ms = 210  # Fixed window size (210ms = 1 fNIRS frame)
        self.eeg_samples_per_frame = 15  # Assuming 15 samples per EEG frame

        # For lazy loading
        self._windows = None

        # Precompute windows if requested
        if precompute_windows:
            self._precompute_windows()
            logger.info(f"Precomputed {len(self._windows)} windows")
        else:
            # Just calculate the number of windows without creating them
            self._calculate_n_windows()
            logger.info(f"Lazy loading enabled, estimated {self.n_windows} windows")

    def _precompute_windows(self):
        """Precompute all windows using the create_windows function."""
        self._windows = create_windows(
            eeg_data=self.eeg_data,
            fnirs_data=self.fnirs_data,
            eeg_timestamps=self.eeg_timestamps,
            fnirs_timestamps=self.fnirs_timestamps,
            return_torch_tensors=(torch is not None)
        )
        self.n_windows = len(self._windows)

    def _calculate_n_windows(self):
        """Estimate the number of windows without creating them all."""
        # This is just an approximation since we can't know about gaps without processing
        has_eeg = self.eeg_data is not None and self.eeg_timestamps is not None and len(self.eeg_timestamps) > 0
        has_fnirs = self.fnirs_data is not None and self.fnirs_timestamps is not None and len(self.fnirs_timestamps) > 0
        
        if not has_eeg and not has_fnirs:
            self.n_windows = 0
            return

        # Estimate number of windows based on available data
        if has_fnirs:
            n_fnirs_frames = self.fnirs_data.shape[0]
            # A rough estimate assuming no gaps
            fnirs_windows = max(0, n_fnirs_frames // self.fnirs_frames_per_window)
        else:
            fnirs_windows = 0
            
        if has_eeg:
            n_eeg_frames = self.eeg_data.shape[0]
            eeg_windows = max(0, n_eeg_frames // self.eeg_frames_per_window)
        else:
            eeg_windows = 0
            
        # Use the min if both are available, otherwise use the one that's available
        if fnirs_windows > 0 and eeg_windows > 0:
            self.n_windows = min(fnirs_windows, eeg_windows)
        else:
            self.n_windows = max(fnirs_windows, eeg_windows)

    def __len__(self) -> int:
        """Return the number of windows in the dataset."""
        # If windows are precomputed, use exact count, otherwise use estimate
        if self._windows is not None:
            return len(self._windows)
        return self.n_windows

    def __getitem__(self, idx: int) -> Dict[str, Union[np.ndarray, 'torch.Tensor', Dict]]:
        """
        Get a specific window by index.

        This either returns a precomputed window or creates it on-demand.

        Args:
            idx: Index of the window to retrieve

        Returns:
            Dictionary with EEG data, fNIRS data, and metadata
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds for dataset with {len(self)} windows")

        # If windows are precomputed, return the precomputed window
        if self._windows is not None:
            return self._windows[idx]
            
        # Lazily compute all windows if we're getting the first one
        # This is more efficient than computing just a single window,
        # as the create_windows function needs to process the entire dataset anyway
        if idx == 0:
            self._precompute_windows()
            return self._windows[idx]
            
        # For other indices, compute all windows - the overhead is similar to computing just one
        self._precompute_windows()
        return self._windows[idx]

    def get_all_windows(self) -> List[Dict]:
        """Return all windows in the dataset.
        
        This is useful for batch operations.
        """
        if self._windows is None:
            self._precompute_windows()
        return self._windows

    def to_dict(self) -> Dict:
        """Convert the dataset to a dictionary format for serialization."""
        return {
            'eeg_data': self.eeg_data,
            'fnirs_data': self.fnirs_data,
            'eeg_timestamps': self.eeg_timestamps,
            'fnirs_timestamps': self.fnirs_timestamps,
            'eeg_frames_per_window': self.eeg_frames_per_window,
            'fnirs_frames_per_window': self.fnirs_frames_per_window,
            'window_size_ms': self.window_size_ms,
            'eeg_samples_per_frame': self.eeg_samples_per_frame
        }

    @classmethod
    def from_dict(cls, data_dict: Dict) -> 'WindowDataset':
        """Create a WindowDataset from a dictionary."""
        dataset = cls(
            eeg_data=data_dict.get('eeg_data'),
            fnirs_data=data_dict.get('fnirs_data'),
            eeg_timestamps=data_dict.get('eeg_timestamps'),
            fnirs_timestamps=data_dict.get('fnirs_timestamps')
        )
        
        # Apply custom parameters if provided
        if 'eeg_frames_per_window' in data_dict:
            dataset.eeg_frames_per_window = data_dict['eeg_frames_per_window']
        if 'fnirs_frames_per_window' in data_dict:
            dataset.fnirs_frames_per_window = data_dict['fnirs_frames_per_window']
        if 'window_size_ms' in data_dict:
            dataset.window_size_ms = data_dict['window_size_ms']
        if 'eeg_samples_per_frame' in data_dict:
            dataset.eeg_samples_per_frame = data_dict['eeg_samples_per_frame']
            
        return dataset