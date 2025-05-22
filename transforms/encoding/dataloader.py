"""
Efficient DataLoader utilities for neural data.

This module provides optimized utilities for loading data from zarr stores
with efficient label preprocessing to avoid repeated conversions during training.
"""

import logging
import numpy as np
import zarr
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Dict, List, Tuple, Union, Optional, Any, Callable

logger = logging.getLogger(__name__)


class EfficientQueryDataset(Dataset):
    """
    Optimized PyTorch Dataset for loading windowed neural data from query zarr stores.
    
    Key improvements over the original QueryDataset:
    1. Pre-converts labels during initialization to avoid repeated conversions
    2. Uses vectorized operations instead of item-by-item processing
    3. Caches expensive operations for better performance
    4. Minimizes debug output for production usage
    5. Provides better error handling and reporting
    """
    
    def __init__(self, 
                 zarr_path: str, 
                 modalities: List[str] = ["eeg", "fnirs"],
                 target_label_map: Optional[Dict[Any, int]] = None,
                 preload_labels: bool = True,
                 verbose: bool = False):
        """
        Initialize the dataset with efficient label preprocessing.
        
        Args:
            zarr_path: Path to query zarr store
            modalities: List of modalities to load ("eeg", "fnirs", or both)
            target_label_map: Optional mapping to standardize labels
            preload_labels: Whether to preload and convert all labels during init
            verbose: Whether to print detailed processing information
        """
        self.zarr_path = zarr_path
        self.modalities = modalities
        self.verbose = verbose
        self.target_label_map = target_label_map or {"closed": 0, "open": 1}
        
        logger.info(f"Initializing EfficientQueryDataset from {zarr_path}")
        
        # Open zarr store
        store_options = {"anon": False}
        self.root = zarr.open_group(store=self.zarr_path, mode="r", 
                                  storage_options=store_options)
        
        # Verify this is a query zarr with sessions
        if "sessions" not in self.root:
            raise ValueError(f"Zarr store at {zarr_path} does not contain a sessions group")
        
        # Get sessions group
        self.sessions_group = self.root["sessions"]
        
        # Extract session IDs
        self.session_ids = list(self.sessions_group.group_keys())
        
        # Build flat index map for all windows across all sessions
        self._build_index_map()
        
        # Pre-load labels if requested (major performance optimization)
        self.cached_labels = None
        if preload_labels:
            self._preload_labels()
            
        logger.info(f"EfficientQueryDataset initialized with {len(self.index_map)} "
                  f"windows across {len(self.session_lengths)} sessions")
        logger.info(f"Modalities: {self.modalities}")

    def _build_index_map(self):
        """
        Build a flat index map for accessing windows.
        
        Maps global dataset indices to (session_id, window_idx) pairs.
        Also validates data shapes and compatibility.
        """
        self.index_map = []
        self.session_lengths = {}
        self.modality_arrays = {}
        
        # Track sessions with issues for reporting
        skipped_sessions = []
        incompatible_shapes = []
        
        for session_id in self.session_ids:
            if self.verbose:
                logger.info(f"Processing session {session_id}")
            
            session_group = self.sessions_group[session_id]
            
            # Track arrays for each modality
            eeg_array_name = None
            fnirs_array_name = None
            window_count = 0
            
            # Find arrays for requested modalities
            
            # Check for EEG data - only look at "eeg" array
            if "eeg" in self.modalities and "eeg" in session_group:
                eeg_array_name = "eeg"
                window_count = session_group["eeg"].shape[0]
                
                # Check EEG shape once per session
                eeg_arr = session_group[eeg_array_name]
                window_shape = eeg_arr.shape[1:]  # Skip first dimension (window count)
                
                if window_shape != (21, 105):
                    incompatible_shapes.append(
                        f"{session_id}: EEG shape {window_shape} (expected (21, 105))"
                    )
                    continue
            
            # Check for fNIRS data
            if "fnirs" in self.modalities and "fnirs" in session_group:
                fnirs_array_name = "fnirs"
                # If no window count yet, set it from fnirs shape
                if window_count == 0:
                    window_count = session_group["fnirs"].shape[0]
            
            # Validation for multi-modal data
            if eeg_array_name and fnirs_array_name:
                eeg_count = session_group[eeg_array_name].shape[0]
                fnirs_count = session_group[fnirs_array_name].shape[0]
                
                if eeg_count != fnirs_count:
                    skipped_sessions.append(
                        f"{session_id}: Mismatched counts - EEG: {eeg_count}, fNIRS: {fnirs_count}"
                    )
                    continue
            
            # Skip sessions with no matching data
            if window_count == 0:
                skipped_sessions.append(
                    f"{session_id}: No data found for requested modalities {self.modalities}"
                )
                continue
            
            # Check for labels (required)
            if "labels" not in session_group:
                skipped_sessions.append(f"{session_id}: No labels found")
                continue
            
            # Verify label count matches window count
            if session_group["labels"].shape[0] != window_count:
                skipped_sessions.append(
                    f"{session_id}: Label count {session_group['labels'].shape[0]} "
                    f"doesn't match window count {window_count}"
                )
                continue
            
            # Store metadata for this session
            self.session_lengths[session_id] = window_count
            self.modality_arrays[session_id] = {
                "eeg": eeg_array_name,
                "fnirs": fnirs_array_name
            }
            
            # Add all indices at once
            for window_idx in range(window_count):
                self.index_map.append((session_id, window_idx))
        
        # Report skipped sessions
        if skipped_sessions:
            logger.warning(f"Skipped {len(skipped_sessions)} sessions due to data issues:")
            for msg in skipped_sessions[:5]:  # Limit to 5 in logs
                logger.warning(f"  {msg}")
            if len(skipped_sessions) > 5:
                logger.warning(f"  ... and {len(skipped_sessions) - 5} more")
        
        # Report incompatible shapes
        if incompatible_shapes:
            logger.warning(f"Skipped {len(incompatible_shapes)} sessions due to incompatible shapes:")
            for msg in incompatible_shapes[:5]:  # Limit to 5 in logs
                logger.warning(f"  {msg}")
            if len(incompatible_shapes) > 5:
                logger.warning(f"  ... and {len(incompatible_shapes) - 5} more")
    
    def _preload_labels(self):
        """
        Preload and preprocess all labels at initialization time.
        
        This is a major optimization to avoid repeated conversions during training.
        """
        logger.info("Preloading and preprocessing all labels")
        
        self.cached_labels = torch.empty(len(self.index_map), dtype=torch.long)
        
        # Process labels in batches for better performance
        batch_size = 1000
        for i in range(0, len(self.index_map), batch_size):
            batch_indices = self.index_map[i:i+batch_size]
            
            # Process each batch of sessions
            for batch_idx, (session_id, window_idx) in enumerate(batch_indices):
                session_group = self.sessions_group[session_id]
                label_data = session_group["labels"][window_idx]
                
                # Vectorized label conversion
                global_idx = i + batch_idx
                self.cached_labels[global_idx] = self._convert_label(label_data)
        
        # Calculate label statistics for debugging
        unique_labels, counts = torch.unique(self.cached_labels, return_counts=True)
        label_stats = dict(zip(unique_labels.tolist(), counts.tolist()))
        
        logger.info(f"Preloaded {len(self.cached_labels)} labels. Distribution: {label_stats}")

    def _convert_label(self, label_data: Any) -> int:
        """
        Convert various label formats to standardized integer.
        
        Args:
            label_data: Label in various possible formats
            
        Returns:
            Standardized integer label
        """
        # Handle 0-D numpy arrays by converting to scalar
        if isinstance(label_data, np.ndarray) and label_data.shape == ():
            label_data = label_data.item()
        
        # Handle string-like labels
        if isinstance(label_data, (bytes, np.bytes_)):
            label_str = label_data.decode('utf-8')
            try:
                return int(label_str)
            except ValueError:
                return self.target_label_map.get(label_str, 0)
                
        elif isinstance(label_data, np.ndarray) and label_data.dtype.kind in ('S', 'a'):
            # Similar handling for ndarray of bytes
            label_str = label_data.item().decode('utf-8')
            try:
                return int(label_str)
            except ValueError:
                return self.target_label_map.get(label_str, 0)
                
        elif isinstance(label_data, str):
            # Direct string
            try:
                return int(label_data)
            except ValueError:
                return self.target_label_map.get(label_data, 0)
                
        elif isinstance(label_data, (int, np.integer)):
            # Already an integer
            return int(label_data)
            
        else:
            # Default fallback
            logger.warning(f"Unhandled label type {type(label_data)}, defaulting to 0")
            return 0
    
    def __len__(self) -> int:
        """Return the total number of windows in the dataset."""
        return len(self.index_map)
    
    def get_session_indices(self) -> Dict[str, List[int]]:
        """
        Get a mapping from session ID to dataset indices.
        
        Returns:
            Dict mapping session IDs to lists of global indices
        """
        result = {sid: [] for sid in self.session_lengths.keys()}
        for i, (sid, _) in enumerate(self.index_map):
            result[sid].append(i)
        return result
    
    def get_feature_shape(self, modality: str) -> tuple:
        """
        Get the feature shape for a modality.
        
        Args:
            modality: The modality to get shape for ("eeg" or "fnirs")
            
        Returns:
            Shape tuple
        """
        if not self.index_map:
            raise ValueError("Dataset is empty, cannot determine feature shape")
            
        # Get the first valid session and window
        session_id, window_idx = self.index_map[0]
        session_group = self.sessions_group[session_id]
        
        if modality == "eeg" and self.modality_arrays[session_id]["eeg"]:
            eeg_array_name = self.modality_arrays[session_id]["eeg"]
            return session_group[eeg_array_name][window_idx].shape
            
        elif modality == "fnirs" and self.modality_arrays[session_id]["fnirs"]:
            fnirs_array_name = self.modality_arrays[session_id]["fnirs"]
            fnirs_data = session_group[fnirs_array_name][window_idx]
            
            # Ensure fnirs data has the right shape for conv1d operations
            if len(fnirs_data.shape) == 1:
                return (fnirs_data.shape[0], 1)
            else:
                return fnirs_data.shape
                
        raise ValueError(f"Modality {modality} not available in dataset")
    
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
        
        # Use cached label if available (major optimization)
        if self.cached_labels is not None:
            label = self.cached_labels[idx]
        else:
            # Fall back to on-demand conversion if needed
            session_id, window_idx = self.index_map[idx]
            session_group = self.sessions_group[session_id]
            label_data = session_group["labels"][window_idx]
            label = torch.tensor(self._convert_label(label_data), dtype=torch.long)
        
        # Get session ID and window index for features
        session_id, window_idx = self.index_map[idx]
        session_group = self.sessions_group[session_id]
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
        
        # For single modality, return the tensor directly
        if len(self.modalities) == 1:
            return features[self.modalities[0]], label
        
        # For multiple modalities, return a dictionary
        return features, label


def create_efficient_data_loaders(
    zarr_path: str,
    batch_size: int = 32,
    modalities: List[str] = ["eeg"],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    shuffle: bool = True,
    num_workers: int = 4,
    seed: int = 42,
    split_by_session: bool = True,
    target_label_map: Optional[Dict[Any, int]] = None,
    filter_label_func: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    preload_labels: bool = True,
    verbose: bool = False
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
    """
    Create optimized PyTorch DataLoaders from a query zarr store.
    
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
        target_label_map: Optional mapping to standardize labels
        filter_label_func: Optional function to filter data by labels
        preload_labels: Whether to preload labels (recommended for performance)
        verbose: Whether to print detailed processing information
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create dataset with optimized label handling
    dataset = EfficientQueryDataset(
        zarr_path=zarr_path, 
        modalities=modalities,
        target_label_map=target_label_map,
        preload_labels=preload_labels,
        verbose=verbose
    )
    
    # Get dataset size
    dataset_size = len(dataset)
    if dataset_size == 0:
        raise ValueError(f"No data found in zarr store at {zarr_path}")
    
    # Apply label filtering if provided
    if filter_label_func is not None and dataset.cached_labels is not None:
        # Use the filter function to get a mask
        mask = filter_label_func(dataset.cached_labels)
        filtered_indices = torch.nonzero(mask).squeeze().tolist()
        
        if not filtered_indices:
            raise ValueError(f"After filtering labels, no samples remain. Check filter_label_func.")
            
        logger.info(f"Applied label filtering: {dataset_size} → {len(filtered_indices)} samples")
        
        # Create a filtered subset
        dataset = torch.utils.data.Subset(dataset, filtered_indices)
        dataset_size = len(dataset)
    
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    
    if split_by_session:
        # -------- session-level split --------------------------------
        # To maintain session boundaries, we need to split by session IDs
        # This prevents data leakage between train/val/test
        if isinstance(dataset, torch.utils.data.Subset):
            # If we're working with a filtered subset, we need to handle this differently
            logger.warning("Label filtering with session-level splitting might not preserve session boundaries perfectly")
            # Simplified approach for filtered subset
            train_size = int(train_ratio * dataset_size)
            val_size = int(val_ratio * dataset_size)
            test_size = dataset_size - train_size - val_size
            
            train_dataset, val_dataset, test_dataset = random_split(
                dataset, [train_size, val_size, test_size]
            )
        else:
            # Standard approach for full dataset
            sess_map = dataset.get_session_indices()
            sess_ids = list(sess_map.keys())
            
            # Random shuffle of session IDs
            rng = torch.Generator().manual_seed(seed)
            sess_ids = [sess_ids[i] for i in torch.randperm(len(sess_ids), generator=rng).tolist()]
            
            n_sess = len(sess_ids)
            n_train = max(1, int(train_ratio * n_sess))
            n_val = max(0, int(val_ratio * n_sess))
            n_test = n_sess - n_train - n_val
            
            # Guarantee at least one session per split when possible
            if n_test == 0 and test_ratio > 0:
                n_test, n_val = 1, max(0, n_val-1)
            
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
            logger.info(f"Session-level split: {n_train} train, {n_val} val, {n_test} test sessions")
            logger.info(f"Window counts: {len(train_dataset) if train_dataset else 0} train, "
                      f"{len(val_dataset) if val_dataset else 0} val, "
                      f"{len(test_dataset) if test_dataset else 0} test windows")
    else:
        # -------- window-level split ------------------------
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
            train_size = int(train_ratio * dataset_size)
            val_size = int(val_ratio * dataset_size)
            test_size = dataset_size - train_size - val_size
            
            train_dataset, val_dataset, test_dataset = random_split(
                dataset, [train_size, val_size, test_size]
            )
    
    # Create data loaders with optimized worker settings
    loader_kwargs = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': True and torch.cuda.is_available(),
        'persistent_workers': num_workers > 0,  # Keep workers alive between iterations
    }
    
    train_loader = DataLoader(
        train_dataset,
        shuffle=shuffle,
        **loader_kwargs
    )
    
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            shuffle=False,  # No need to shuffle validation data
            **loader_kwargs
        )
    
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            shuffle=False,  # No need to shuffle test data
            **loader_kwargs
        )
    
    return train_loader, val_loader, test_loader


# Utility functions for common label filtering needs

def filter_binary_labels(labels: torch.Tensor) -> torch.Tensor:
    """
    Filter for only binary labels (0 and 1).
    
    Args:
        labels: Tensor of labels
        
    Returns:
        Boolean mask for valid labels
    """
    return (labels == 0) | (labels == 1)


def filter_labels_in_range(min_label: int, max_label: int):
    """
    Create a function to filter labels within a specific range.
    
    Args:
        min_label: Minimum label value (inclusive)
        max_label: Maximum label value (inclusive)
        
    Returns:
        Filter function that takes a tensor and returns a boolean mask
    """
    def filter_func(labels: torch.Tensor) -> torch.Tensor:
        return (labels >= min_label) & (labels <= max_label)
    return filter_func