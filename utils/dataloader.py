"""
PyTorch DataLoader utilities for neural window data.

This module provides utilities for loading data from query zarr stores (outputs
of t4A_query_v0.py) into PyTorch DataLoaders for training machine learning models.
"""

import logging
import numpy as np
import zarr
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Dict, List, Tuple, Union, Optional

logger = logging.getLogger(__name__)


class QueryDataset(Dataset):
    """
    PyTorch Dataset for loading windowed neural data from any query zarr store.
    
    Loads windows from the output of t4A_query_v0.py, which contains data from
    multiple sessions with neural windows (EEG and/or fNIRS) and labels.
    
    Expected data shapes (but handles variations):
    - EEG: (time, eeg_channel, eeg_sample) 
    - fNIRS: (time, fnirs_channel) or (time, fnirs_channel, 1)
    """
    
    def __init__(self, 
                 zarr_path: str, 
                 modalities: List[str] = ["eeg", "fnirs"]):
        """
        Initialize the dataset.
        
        Args:
            zarr_path: Path to query zarr store
            modalities: List of modalities to load ("eeg", "fnirs", or both)
        """
        self.zarr_path = zarr_path
        self.modalities = modalities
        
        # Open zarr store
        self.root = zarr.open_group(store=self.zarr_path, mode="r", 
                                   storage_options={"anon": False})
        
        # Verify this is a query zarr with sessions and metadata
        if "sessions" not in self.root:
            raise ValueError(f"Zarr store at {zarr_path} does not contain a sessions group")
        
        # Get sessions group
        self.sessions_group = self.root["sessions"]
        
        # Extract session IDs
        self.session_ids = list(self.sessions_group.group_keys())
        
        # Build flat index map for all windows across all sessions
        self._build_index_map()
        
        print(f"QueryDataset initialized with {len(self.index_map)} total windows across {len(self.session_ids)} sessions")
        print(f"Modalities: {self.modalities}")

    def _build_index_map(self):
        """
        Build a flat index map for accessing windows.
        
        Maps global dataset indices to (session_id, window_idx) pairs.
        """
        self.index_map = []
        self.session_lengths = {}
        self.window_array_names = {}
        self.modality_arrays = {}
        
        for session_id in self.session_ids:
            print(f"QueryDataset: Processing session {session_id}")
            session_group = self.sessions_group[session_id]
            
            # Track arrays for each modality
            eeg_array_name = None
            fnirs_array_name = None
            window_count = 0
            
            # Find arrays for requested modalities
            
            # Check for EEG data - ONLY look at "eeg" array, NEVER at "windows"
            if "eeg" in self.modalities and "eeg" in session_group:
                eeg_array_name = "eeg"
                window_count = session_group["eeg"].shape[0]
            
            # Check for fNIRS data
            if "fnirs" in self.modalities and "fnirs" in session_group:
                fnirs_array_name = "fnirs"
                # If no window count yet, set it from fnirs shape
                if window_count == 0:
                    window_count = session_group["fnirs"].shape[0]
            
            # Determine the primary array for window count
            # For cases with both modalities, validate they have the same count
            if eeg_array_name and fnirs_array_name:
                eeg_count = session_group[eeg_array_name].shape[0]
                fnirs_count = session_group[fnirs_array_name].shape[0]
                
                if eeg_count != fnirs_count:
                    print(
                        f"Skipping session {session_id}: Mismatched counts - " 
                        f"EEG: {eeg_count}, fNIRS: {fnirs_count}"
                    )
                    continue
            
            # Skip sessions with no matching data for requested modalities
            if window_count == 0:
                print(f"Skipping session {session_id}: No data found for requested modalities {self.modalities}")
                continue
            
            # Check for labels (required)
            if "labels" not in session_group:
                print(f"Skipping session {session_id}: No labels found")
                continue
            
            # Verify label count matches window count
            if session_group["labels"].shape[0] != window_count:
                print(
                    f"Skipping session {session_id}: Label count {session_group['labels'].shape[0]} "
                    f"doesn't match window count {window_count}"
                )
                continue
            
            # Check EEG shape once per session using array metadata
            if "eeg" in self.modalities and eeg_array_name:
                eeg_arr = session_group[eeg_array_name]
                window_shape = eeg_arr.shape[1:]  # Skip first dimension (window count)
                
                if window_shape != (21, 105):
                    print(
                        f"{session_id}: skipping session with EEG shape "
                        f"{window_shape} (expected (21, 105))."
                    )
                    # Skip entire session
                    continue
            
            # Store metadata for this session
            self.session_lengths[session_id] = window_count
            self.modality_arrays[session_id] = {
                "eeg": eeg_array_name,
                "fnirs": fnirs_array_name
            }
            
            # Add all indices at once, now that we've verified the shape at session level
            for window_idx in range(window_count):
                self.index_map.append((session_id, window_idx))
    
    def __len__(self) -> int:
        """Return the total number of windows in the dataset."""
        return len(self.index_map)
    
    def sessions(self) -> List[str]:
        """Return the list of session-IDs in the same order
        that indices appear in `index_map`."""
        return self.session_ids.copy()

    def indices_by_session(self) -> Dict[str, List[int]]:
        """Map each session-ID to the list of dataset indices
        that belong to it."""
        out = {sid: [] for sid in self.session_ids}
        for global_idx, (sid, _) in enumerate(self.index_map):
            out[sid].append(global_idx)
        return out
    
    def __getitem__(self, idx: int) -> Tuple[Union[Dict[str, torch.Tensor], torch.Tensor], torch.Tensor]:
        """
        Get a window and its label.
        
        Args:
            idx: Global dataset index
            
        Returns:
            Tuple of (features, label) where features is either a tensor (single modality)
            or a dict of tensors (multiple modalities)
        """
        if idx < 0 or idx >= len(self.index_map):
            raise IndexError(f"Index {idx} out of bounds for dataset with {len(self.index_map)} items")
        
        # Get session ID and window index
        session_id, window_idx = self.index_map[idx]
        
        # Get session group
        session_group = self.sessions_group[session_id]
        
        # Get array names for this session's modalities
        modality_arrays = self.modality_arrays[session_id]
        
        # Extract features based on requested modalities
        features = {}
        
        # Handle EEG data
        if "eeg" in self.modalities and modality_arrays["eeg"]:
            eeg_array_name = modality_arrays["eeg"]
            eeg_data = session_group[eeg_array_name][window_idx]
            features["eeg"] = torch.tensor(eeg_data, dtype=torch.float32)
        
        # Handle fNIRS data
        if "fnirs" in self.modalities and modality_arrays["fnirs"]:
            fnirs_array_name = modality_arrays["fnirs"]
            fnirs_data = session_group[fnirs_array_name][window_idx]
            
            # Ensure fnirs data has the right shape for conv1d operations
            if len(fnirs_data.shape) == 1:
                # If 1D, reshape to (channels, 1)
                fnirs_data = fnirs_data.reshape(-1, 1)
            elif len(fnirs_data.shape) > 2:
                # If more than 2D, reshape to (channels, -1)
                channels = fnirs_data.shape[0]
                fnirs_data = fnirs_data.reshape(channels, -1)
            
            features["fnirs"] = torch.tensor(fnirs_data, dtype=torch.float32)
        
        # Extract label
        label_data = session_group["labels"][window_idx]
        print(f"Raw label_data: {label_data}, type: {type(label_data)}, shape: {getattr(label_data, 'shape', None)}")
        
        # CRITICAL FIX: Handle 0-D numpy arrays by converting to scalar
        if isinstance(label_data, np.ndarray) and label_data.shape == ():
            print(f"Converting 0-D array to scalar: {label_data} -> {label_data.item()}")
            label_data = label_data.item()
        
        # Get label map from root attributes if available
        label_map = None
        if hasattr(self.root, 'attrs') and 'label_map' in self.root.attrs:
            label_map = self.root.attrs.get("label_map", {"closed": 0, "open": 1})
        else:
            label_map = {"closed": 0, "open": 1}
        
        # Handle different label types
        if isinstance(label_data, (bytes, np.bytes_)):
            label_str = label_data.decode('utf-8')
            try:
                label_data = int(label_str)
            except ValueError:
                label_data = label_map.get(label_str, 0)
        elif isinstance(label_data, np.ndarray) and label_data.dtype.kind in ('S', 'a'):
            # Similar handling for ndarray of bytes
            label_str = label_data.item().decode('utf-8')
            try:
                label_data = int(label_str)
            except ValueError:
                label_data = label_map.get(label_str, 0)
        elif isinstance(label_data, str):
            # Direct string
            try:
                label_data = int(label_data)
            except ValueError:
                label_data = label_map.get(label_data, 0)
        elif isinstance(label_data, (int, np.integer)):
            # Already an integer
            label_data = int(label_data)
            print(f"Integer label: {label_data}")
        else:
            # Default fallback
            print(f"WARNING: Unhandled label type {type(label_data)}, defaulting to 0")
            label_data = 0
        
        # Convert label to tensor
        label = torch.tensor(label_data, dtype=torch.long)
        print(f"Final label tensor: {label}")
        
        # For single modality, return the tensor directly
        if len(self.modalities) == 1:
            return features[self.modalities[0]], label
        
        # For multiple modalities, return a dictionary
        return features, label


def create_data_loaders(
    zarr_path: str,
    batch_size: int = 32,
    modalities: List[str] = ["eeg", "fnirs"],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    shuffle: bool = True,
    num_workers: int = 4,
    seed: int = 42,
    split_by_session: bool = False
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
    """
    Create PyTorch DataLoaders from a query zarr store.
    
    Args:
        zarr_path: Path to query zarr store (S3 URI or local path)
        batch_size: Batch size for DataLoader
        modalities: List of modalities to load ("eeg", "fnirs", or both)
        train_ratio: Fraction of data for training
        val_ratio: Fraction of data for validation
        test_ratio: Fraction of data for testing
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes for DataLoader
        seed: Random seed for reproducible splits
        split_by_session: If True, maintain session boundaries when splitting 
                         (all windows from a session stay in the same split)
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
        
    Example:
        >>> train_loader, val_loader, test_loader = create_data_loaders(
        ...     zarr_path="s3://conduit-data-dev/processed/queries/eye_neural.zarr",
        ...     batch_size=32,
        ...     modalities=["eeg"],
        ...     train_ratio=0.8,
        ...     val_ratio=0.2,
        ...     test_ratio=0.0,
        ...     split_by_session=True  # Use session-level splitting
        ... )
    """
    # Create dataset
    dataset = QueryDataset(zarr_path=zarr_path, modalities=modalities)
    
    # Get dataset size
    dataset_size = len(dataset)
    if dataset_size == 0:
        raise ValueError(f"No data found in zarr store at {zarr_path}")
    
    # Calculate split sizes
    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    test_size = dataset_size - train_size - val_size
    
    # Ensure minimum sizes
    if train_size == 0:
        raise ValueError(f"Training set size is 0 with {dataset_size} total items and train_ratio={train_ratio}")
    if val_ratio > 0 and val_size == 0:
        raise ValueError(f"Validation set size is 0 with {dataset_size} total items and val_ratio={val_ratio}")
    if test_ratio > 0 and test_size == 0:
        raise ValueError(f"Test set size is 0 with {dataset_size} total items and test_ratio={test_ratio}")
    
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    
    if split_by_session:
        # -------- session-level split --------------------------------
        sess_map = dataset.indices_by_session()
        sess_ids = list(sess_map)
        rng = torch.Generator().manual_seed(seed)
        sess_ids = [sess_ids[i] for i in torch.randperm(len(sess_ids), generator=rng).tolist()]
        
        n_sess = len(sess_ids)
        n_train = max(1, int(train_ratio * n_sess))
        n_val = max(0, int(val_ratio * n_sess))
        n_test = n_sess - n_train - n_val
        
        # guarantee at least one session per split when possible
        if n_test == 0 and test_ratio > 0: n_test, n_val = 1, max(0, n_val-1)
        
        train_sess = sess_ids[:n_train]
        val_sess = sess_ids[n_train:n_train+n_val]
        test_sess = sess_ids[n_train+n_val:]
        
        def subset(sids):
            flat = [idx for sid in sids for idx in sess_map[sid]]
            return torch.utils.data.Subset(dataset, flat)
        
        train_dataset = subset(train_sess)
        val_dataset = subset(val_sess) if val_sess else None
        test_dataset = subset(test_sess) if test_sess else None
        
        # Log session distribution
        print(f"Session-level split: {n_train} train, {n_val} val, {n_test} test sessions")
        print(f"Window counts: {len(train_dataset) if train_dataset else 0} train, "
              f"{len(val_dataset) if val_dataset else 0} val, "
              f"{len(test_dataset) if test_dataset else 0} test windows")
    else:
        # -------- original window-level split ------------------------
        if val_size == 0 and test_size == 0:
            # Only training data
            train_dataset = dataset
            val_dataset = None
            test_dataset = None
        elif test_size == 0:
            # Training and validation only
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
            test_dataset = None
        else:
            # Full split
            train_dataset, val_dataset, test_dataset = random_split(
                dataset, [train_size, val_size, test_size]
            )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    
    return train_loader, val_loader, test_loader