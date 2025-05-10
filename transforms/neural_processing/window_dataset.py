"""
PyTorch Dataset for synchronized EEG and fNIRS windows derived from pre-aligned NumPy arrays.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import logging
from typing import Dict, Tuple, Union, Optional

logger = logging.getLogger(__name__)

class WindowDataset(Dataset):
    """
    A PyTorch Dataset that provides synchronized windows of EEG and fNIRS data.

    It operates on a pre-defined span where both modalities are considered valid,
    as determined by 'both_start_idx' and 'both_end_idx'.
    """
    def __init__(self,
                 eeg_windows_data_arr: np.ndarray,
                 fnirs_windows_data_arr: np.ndarray,
                 eeg_validity_mask_arr: np.ndarray,
                 fnirs_validity_mask_arr: np.ndarray,
                 real_eeg_timestamps_arr: np.ndarray,
                 real_fnirs_timestamps_arr: np.ndarray,
                 main_clock_timestamps_array: np.ndarray,
                 eeg_start_idx: int, # Index in the master arrays
                 fnirs_start_idx: int, # Index in the master arrays
                 eeg_end_idx: int,     # Exclusive end index
                 fnirs_end_idx: int,   # Exclusive end index
                 both_start_idx: int,  # Inclusive start index for this dataset's view
                 both_end_idx: int,    # Exclusive end index for this dataset's view
                 return_torch_tensors: bool = True):
        """
        Initializes the WindowDataset.

        Args:
            eeg_windows_data_arr: (total_windows, num_eeg_channels, 105_samples)
            fnirs_windows_data_arr: (total_windows, num_fnirs_channels, 1_sample)
            eeg_validity_mask_arr: (total_windows,) bool
            fnirs_validity_mask_arr: (total_windows,) bool
            real_eeg_timestamps_arr: (total_windows,) float, actual EEG start ts
            real_fnirs_timestamps_arr: (total_windows,) float, actual fNIRS start ts
            main_clock_timestamps_array: (total_windows,) float, ideal start ts
            eeg_start_idx: First index in master arrays where valid EEG data might start.
            fnirs_start_idx: First index in master arrays where valid fNIRS data might start.
            eeg_end_idx: Index after the last in master arrays where valid EEG data might end.
            fnirs_end_idx: Index after the last in master arrays where valid fNIRS data might end.
            both_start_idx: The starting index in the master arrays for this dataset's synchronized view.
            both_end_idx: The exclusive end index in the master arrays for this dataset's synchronized view.
            return_torch_tensors: If True, __getitem__ returns torch tensors.
        """

        if not (isinstance(both_start_idx, int) and isinstance(both_end_idx, int) and \
                both_start_idx != -1 and both_end_idx != -1 and both_start_idx < both_end_idx):
            raise ValueError(
                f"Invalid 'both_start_idx' ({both_start_idx}) or 'both_end_idx' ({both_end_idx}). "
                f"No valid span where both EEG and fNIRS data are present. Cannot create dataset."
            )

        self.eeg_data = eeg_windows_data_arr
        self.fnirs_data = fnirs_windows_data_arr
        self.eeg_validity = eeg_validity_mask_arr
        self.fnirs_validity = fnirs_validity_mask_arr
        self.real_eeg_ts = real_eeg_timestamps_arr
        self.real_fnirs_ts = real_fnirs_timestamps_arr
        self.main_clock_ts = main_clock_timestamps_array
        
        # These are stored for potential reference but not directly used by len/getitem for this dataset's view
        self.master_eeg_start_idx = eeg_start_idx
        self.master_fnirs_start_idx = fnirs_start_idx
        self.master_eeg_end_idx = eeg_end_idx
        self.master_fnirs_end_idx = fnirs_end_idx

        self.slice_start = both_start_idx
        self.slice_end = both_end_idx
        self.return_torch_tensors = return_torch_tensors
        
        self._length = self.slice_end - self.slice_start
        
        logger.info(f"WindowDataset initialized. Effective length (both modalities valid): {self._length} windows. "
                    f"Slicing master arrays from index {self.slice_start} to {self.slice_end-1}.")

    def __len__(self) -> int:
        """Returns the number of windows where both EEG and fNIRS are valid."""
        return self._length

    def __getitem__(self, idx: int) -> Dict[str, Union[np.ndarray, torch.Tensor, bool, float, int]]:
        """
        Retrieves a synchronized EEG and fNIRS window.

        Args:
            idx: Index relative to the start of the 'both_valid' span.

        Returns:
            A dictionary containing 'eeg' data, 'fnirs' data, and 'metadata'.
        """
        if not (0 <= idx < self._length):
            raise IndexError(f"Index {idx} out of bounds for WindowDataset of length {self._length}.")

        actual_array_idx = self.slice_start + idx

        eeg_item = self.eeg_data[actual_array_idx]
        fnirs_item = self.fnirs_data[actual_array_idx]

        metadata = {
            'eeg_valid': self.eeg_validity[actual_array_idx],
            'fnirs_valid': self.fnirs_validity[actual_array_idx],
            'real_eeg_start_ts': self.real_eeg_ts[actual_array_idx],
            'real_fnirs_start_ts': self.real_fnirs_ts[actual_array_idx],
            'main_clock_ts': self.main_clock_ts[actual_array_idx],
            'original_array_idx': actual_array_idx, # Index in the full NumPy arrays
            'dataset_idx': idx # Index relative to this dataset's start
        }
        
        # All data within this dataset's slice should ideally have both eeg_valid and fnirs_valid as True
        # due to the constructor check on both_start_idx and both_end_idx.
        # If they were derived from eeg_validity & fnirs_validity, this should hold.

        if self.return_torch_tensors:
            eeg_item = torch.from_numpy(eeg_item.copy()).float()
            fnirs_item = torch.from_numpy(fnirs_item.copy()).float()

        return {
            'eeg': eeg_item,
            'fnirs': fnirs_item,
            'metadata': metadata
        }

