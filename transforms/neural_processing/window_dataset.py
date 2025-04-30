"""
WindowDataset class for storing and accessing windowed neural data.
"""

import numpy as np


class WindowDataset:
    """
    Dataset class for storing and accessing windowed neural data.
    """
    
    def __init__(self, eeg_windows=None, fnirs_windows=None, metadata=None):
        """
        Initialize a WindowDataset.
        
        Args:
            eeg_windows: List of EEG data windows
            fnirs_windows: List of fNIRS data windows
            metadata: List of metadata dictionaries for each window
        """
        self.eeg_windows = eeg_windows or []
        self.fnirs_windows = fnirs_windows or []
        self.metadata = metadata or []
        
        # Validate that all lists have the same length
        if len(self.eeg_windows) != len(self.fnirs_windows) or len(self.eeg_windows) != len(self.metadata):
            raise ValueError("All window lists must have the same length")
    
    def __len__(self):
        """Return the number of windows in the dataset."""
        return len(self.metadata)
    
    def __getitem__(self, idx):
        """Get a specific window by index."""
        return {
            'eeg': self.eeg_windows[idx],
            'fnirs': self.fnirs_windows[idx],
            'metadata': self.metadata[idx]
        }
    
    def to_dict(self):
        """Convert the dataset to a dictionary format."""
        return {
            'eeg_windows': self.eeg_windows,
            'fnirs_windows': self.fnirs_windows,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data_dict):
        """Create a WindowDataset from a dictionary."""
        return cls(
            eeg_windows=data_dict.get('eeg_windows', []),
            fnirs_windows=data_dict.get('fnirs_windows', []),
            metadata=data_dict.get('metadata', [])
        )