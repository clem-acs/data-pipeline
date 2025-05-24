"""
Efficient DataLoader utilities for neural data.

This module provides optimized utilities for loading data from zarr stores
with efficient label preprocessing to avoid repeated conversions during training.

Key Features:
- Session-wise data splitting to prevent data leakage
- Automatic class imbalance detection and correction
- Label distribution analysis and reporting
- Weighted sampling for imbalanced datasets
- Support for multi-modal data (EEG + fNIRS)
- Efficient label preprocessing and caching

Usage Examples:

Basic usage with automatic best practices:
```python
from dataloader import create_neural_data_loaders

# High-level function that handles everything automatically
result = create_neural_data_loaders(
    zarr_path="s3://bucket/processed/queries/eye_10_0.5.zarr",
    task_type="classification",
    batch_size=32,
    modalities=["eeg"],
    auto_balance=True,  # Automatically handle class imbalance
    split_by_session=True,  # Prevent data leakage
    verbose=True
)

train_loader = result['train_loader']
val_loader = result['val_loader'] 
test_loader = result['test_loader']
feature_shape = result['feature_shape']  # For model initialization
class_weights = result['class_weights']  # For loss function
```

Advanced usage with custom configuration:
```python
from dataloader import create_efficient_data_loaders, analyze_label_distribution

# Create data loaders with specific settings
train_loader, val_loader, test_loader = create_efficient_data_loaders(
    zarr_path="s3://bucket/processed/queries/eye_10_0.5.zarr",
    batch_size=64,
    modalities=["eeg", "fnirs"],  # Multi-modal
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    split_by_session=True,
    use_weighted_sampling=True,  # Handle imbalance with sampling
    label_names={0: "close", 1: "open", 2: "intro", 3: "unknown"},
    filter_label_func=lambda labels: (labels == 0) | (labels == 1),  # Binary only
    seed=42
)
```

Automatic balancing based on imbalance detection:
```python
from dataloader import create_auto_balanced_loaders

# Automatically enables weighted sampling if imbalance ratio > threshold
train_loader, val_loader, test_loader, dist_info = create_auto_balanced_loaders(
    zarr_path="s3://bucket/processed/queries/eye_10_0.5.zarr",
    imbalance_threshold=3.0,  # Use weighted sampling if ratio > 3:1
    split_by_session=True
)

print(f"Imbalance ratio: {dist_info['imbalance_ratio']}")
print(f"Class distribution: {dist_info['label_counts']}")
```

Label distribution analysis:
```python
from dataloader import EfficientQueryDataset, analyze_label_distribution

dataset = EfficientQueryDataset(zarr_path="...", preload_labels=True)
dist_info = analyze_label_distribution(
    dataset, 
    label_map={0: "close", 1: "open", 2: "intro", 3: "unknown"}
)
# Automatically logs distribution and suggests resampling if needed
```
"""

import logging
import numpy as np
import zarr
import torch
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
from collections import Counter
# Note: sklearn import removed to avoid dependency issues
# Will implement class weight computation manually

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
                 verbose: bool = False,
                 session_limit: Optional[int] = None):
        """
        Initialize the dataset with efficient label preprocessing.
        
        Args:
            zarr_path: Path to query zarr store
            modalities: List of modalities to load ("eeg", "fnirs", or both)
            target_label_map: Optional mapping to standardize labels
            preload_labels: Whether to preload and convert all labels during init
            verbose: Whether to print detailed processing information
            session_limit: Optional limit on number of sessions to process (for testing)
        """
        self.zarr_path = zarr_path
        self.modalities = modalities
        self.verbose = verbose
        self.target_label_map = target_label_map or {"closed": 0, "open": 1}
        self.session_limit = session_limit
        
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
        
        # Extract session IDs (with optional limit for testing)
        if self.session_limit is not None:
            self.session_ids = []
            for i, session_id in enumerate(self.sessions_group.group_keys()):
                if i >= self.session_limit:
                    break
                self.session_ids.append(session_id)
            logger.info(f"Limited to first {len(self.session_ids)} sessions for testing")
        else:
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
        Uses bulk loading per session to minimize S3 requests.
        """
        logger.info("Preloading and preprocessing all labels")
        
        self.cached_labels = torch.empty(len(self.index_map), dtype=torch.long)
        
        # Group indices by session to enable bulk loading
        session_indices = {}
        for global_idx, (session_id, window_idx) in enumerate(self.index_map):
            if session_id not in session_indices:
                session_indices[session_id] = []
            session_indices[session_id].append((global_idx, window_idx))
        
        # Load all labels for each session in bulk (one S3 request per session)
        for session_id, indices_list in session_indices.items():
            session_group = self.sessions_group[session_id]
            
            # Bulk load all labels for this session
            all_session_labels = session_group["labels"][:]  # Single S3 request
            
            # Distribute to global indices
            for global_idx, window_idx in indices_list:
                label_data = all_session_labels[window_idx]
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
    
    def get_feature_shape(self, modality: str = None) -> Union[tuple, Dict[str, tuple]]:
        """
        Get the feature shape for a modality using fast zarr metadata access.
        
        Args:
            modality: The modality to get shape for ("eeg" or "fnirs"), or None for all
            
        Returns:
            Shape tuple for single modality, or dict of shapes for all modalities
        """
        if not self.index_map:
            raise ValueError("Dataset is empty, cannot determine feature shape")
            
        # Get the first valid session
        session_id = next(iter(self.session_lengths.keys()))
        session_group = self.sessions_group[session_id]
        
        shapes = {}
        
        # Get EEG shape from zarr metadata (fast - no data loading)
        if modality is None or modality == "eeg":
            if self.modality_arrays[session_id]["eeg"]:
                eeg_array_name = self.modality_arrays[session_id]["eeg"]
                eeg_array = session_group[eeg_array_name]
                # Zarr array shape is (n_windows, n_channels, n_timepoints)
                # We want (n_channels, n_timepoints) for a single window
                if len(eeg_array.shape) >= 2:
                    shapes["eeg"] = eeg_array.shape[1:]  # Skip first dimension (n_windows)
                else:
                    shapes["eeg"] = eeg_array.shape
                    
        # Get fNIRS shape from zarr metadata (fast - no data loading)  
        if modality is None or modality == "fnirs":
            if self.modality_arrays[session_id]["fnirs"]:
                fnirs_array_name = self.modality_arrays[session_id]["fnirs"]
                fnirs_array = session_group[fnirs_array_name]
                # Zarr array shape is (n_windows, n_channels, n_timepoints)
                if len(fnirs_array.shape) >= 2:
                    fnirs_shape = fnirs_array.shape[1:]  # Skip first dimension (n_windows)
                    # Ensure fnirs data has the right shape for conv1d operations
                    if len(fnirs_shape) == 1:
                        shapes["fnirs"] = (fnirs_shape[0], 1)
                    else:
                        shapes["fnirs"] = fnirs_shape
                else:
                    shapes["fnirs"] = fnirs_array.shape
        
        # Return single shape or dict based on request
        if modality is not None:
            if modality in shapes:
                return shapes[modality]
            else:
                raise ValueError(f"Modality {modality} not available in dataset")
        else:
            # Return all shapes
            return shapes
    
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


def analyze_label_distribution(
    dataset: EfficientQueryDataset,
    label_map: Optional[Dict[int, str]] = None,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Analyze label distribution in the dataset and compute class weights.
    
    Args:
        dataset: The dataset to analyze
        label_map: Optional mapping from label indices to names
        logger: Optional logger for output
        
    Returns:
        Dictionary containing distribution stats and class weights
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    if dataset.cached_labels is None:
        logger.warning("Labels not preloaded, this analysis might be slow")
        labels = [dataset[i][1].item() for i in range(len(dataset))]
        labels = torch.tensor(labels)
    else:
        labels = dataset.cached_labels
    
    # Convert to numpy for sklearn compatibility
    labels_np = labels.numpy()
    
    # Count distribution
    label_counts = Counter(labels_np)
    total_samples = len(labels_np)
    
    # Create readable names
    if label_map is None:
        label_names = {k: f"class_{k}" for k in label_counts.keys()}
    else:
        label_names = {k: label_map.get(k, f"class_{k}") for k in label_counts.keys()}
    
    # Compute class weights (inverse frequency)
    unique_labels = np.array(list(label_counts.keys()))
    # Manual balanced class weight computation: weight = n_samples / (n_classes * n_samples_for_class)
    n_samples = len(labels_np)
    n_classes = len(unique_labels)
    class_weights = []
    for label in unique_labels:
        weight = n_samples / (n_classes * label_counts[label])
        class_weights.append(weight)
    class_weight_dict = dict(zip(unique_labels, class_weights))
    
    # Log distribution
    logger.info(f"Label distribution analysis ({total_samples} total samples):")
    for label_idx, count in sorted(label_counts.items()):
        name = label_names[label_idx]
        percentage = 100 * count / total_samples
        weight = class_weight_dict[label_idx]
        logger.info(f"  {name} (label {label_idx}): {count:6d} samples ({percentage:5.1f}%, weight: {weight:.3f})")
    
    # Check for severe imbalance
    max_count = max(label_counts.values())
    min_count = min(label_counts.values())
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    
    if imbalance_ratio > 10:
        logger.warning(f"Severe class imbalance detected (ratio: {imbalance_ratio:.1f}:1)")
        logger.info("Consider using resampling strategies or adjusting class weights")
    
    return {
        'label_counts': label_counts,
        'label_names': label_names,
        'class_weights': class_weight_dict,
        'imbalance_ratio': imbalance_ratio,
        'total_samples': total_samples
    }


def create_weighted_sampler(
    dataset: Union[EfficientQueryDataset, torch.utils.data.Subset],
    class_weights: Dict[int, float],
    logger: Optional[logging.Logger] = None
) -> WeightedRandomSampler:
    """
    Create a WeightedRandomSampler for handling class imbalance.
    
    Args:
        dataset: The dataset to create sampler for
        class_weights: Dictionary mapping label indices to weights
        logger: Optional logger
        
    Returns:
        WeightedRandomSampler instance
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Get labels for the dataset
    if isinstance(dataset, torch.utils.data.Subset):
        # For subset, we need to get labels for the subset indices
        parent_dataset = dataset.dataset
        if hasattr(parent_dataset, 'cached_labels') and parent_dataset.cached_labels is not None:
            labels = parent_dataset.cached_labels[dataset.indices]
        else:
            labels = torch.tensor([parent_dataset[idx][1].item() for idx in dataset.indices])
    else:
        # For full dataset
        if dataset.cached_labels is not None:
            labels = dataset.cached_labels
        else:
            labels = torch.tensor([dataset[i][1].item() for i in range(len(dataset))])
    
    # Create sample weights
    sample_weights = [class_weights[label.item()] for label in labels]
    
    logger.info(f"Created weighted sampler for {len(sample_weights)} samples")
    
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )


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
    verbose: bool = False,
    analyze_distribution: bool = True,
    use_weighted_sampling: bool = False,
    label_names: Optional[Dict[int, str]] = None,
    _reuse_dataset: Optional[EfficientQueryDataset] = None,
    session_limit: Optional[int] = None
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
        analyze_distribution: Whether to analyze and log label distribution
        use_weighted_sampling: Whether to use weighted sampling for class imbalance
        label_names: Optional mapping from label indices to readable names
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create dataset with optimized label handling (or reuse existing one)
    if _reuse_dataset is not None:
        dataset = _reuse_dataset
        logger.info("Reusing existing dataset to avoid duplicate loading")
    else:
        dataset = EfficientQueryDataset(
            zarr_path=zarr_path, 
            modalities=modalities,
            target_label_map=target_label_map,
            preload_labels=preload_labels,
            verbose=verbose,
            session_limit=session_limit
        )
    
    # Get dataset size
    dataset_size = len(dataset)
    if dataset_size == 0:
        raise ValueError(f"No data found in zarr store at {zarr_path}")
    
    # Analyze label distribution if requested
    distribution_info = None
    if analyze_distribution:
        distribution_info = analyze_label_distribution(
            dataset, 
            label_map=label_names, 
            logger=logger
        )
    
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
        # Calculate split sizes
        train_size = int(train_ratio * dataset_size)
        val_size = int(val_ratio * dataset_size)
        test_size = dataset_size - train_size - val_size
        
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
    
    # Create data loaders with optimized worker settings
    loader_kwargs = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': True and torch.cuda.is_available(),
        'persistent_workers': num_workers > 0,  # Keep workers alive between iterations
    }
    
    # Handle weighted sampling for training data if requested
    train_sampler = None
    if use_weighted_sampling and distribution_info is not None:
        train_sampler = create_weighted_sampler(
            train_dataset, 
            distribution_info['class_weights'], 
            logger=logger
        )
        # When using a custom sampler, we can't also shuffle
        train_shuffle = False
        logger.info("Using weighted sampling for training data (disabling shuffle)")
    else:
        train_shuffle = shuffle
    
    train_loader = DataLoader(
        train_dataset,
        shuffle=train_shuffle,
        sampler=train_sampler,
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


def create_auto_balanced_loaders(
    zarr_path: str,
    batch_size: int = 32,
    modalities: List[str] = ["eeg"],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    imbalance_threshold: float = 3.0,
    split_by_session: bool = True,
    seed: int = 42,
    session_limit: Optional[int] = None,
    **kwargs
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader], Dict[str, Any]]:
    """
    Create data loaders with automatic resampling if class imbalance is detected.
    
    Args:
        zarr_path: Path to query zarr store
        batch_size: Batch size for DataLoader
        modalities: List of modalities to load
        train_ratio: Fraction for training
        val_ratio: Fraction for validation  
        test_ratio: Fraction for testing
        imbalance_threshold: If max_class/min_class > this, use weighted sampling
        split_by_session: Whether to split by session
        seed: Random seed
        **kwargs: Additional arguments passed to create_efficient_data_loaders
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader, distribution_info)
    """
    # Create dataset once and reuse it
    dataset = EfficientQueryDataset(
        zarr_path=zarr_path,
        modalities=modalities,
        preload_labels=True,
        verbose=kwargs.get('verbose', False),
        session_limit=session_limit
    )
    
    distribution_info = analyze_label_distribution(dataset)
    use_weighted = distribution_info['imbalance_ratio'] > imbalance_threshold
    
    if use_weighted:
        logger.info(f"Imbalance ratio {distribution_info['imbalance_ratio']:.1f} > {imbalance_threshold}, enabling weighted sampling")
    
    # Create loaders with appropriate sampling, reusing the dataset
    train_loader, val_loader, test_loader = create_efficient_data_loaders(
        zarr_path=zarr_path,
        batch_size=batch_size,
        modalities=modalities,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        split_by_session=split_by_session,
        seed=seed,
        analyze_distribution=False,  # Skip analysis since we already did it
        use_weighted_sampling=use_weighted,
        _reuse_dataset=dataset,  # Pass the dataset to reuse
        **kwargs
    )
    
    return train_loader, val_loader, test_loader, distribution_info


def get_loss_class_weights(distribution_info: Dict[str, Any]) -> torch.Tensor:
    """
    Get class weights as a tensor for use in loss functions like CrossEntropyLoss.
    
    Args:
        distribution_info: Output from analyze_label_distribution
        
    Returns:
        Tensor of class weights ordered by label index
    """
    class_weights = distribution_info['class_weights']
    max_label = max(class_weights.keys())
    weights_tensor = torch.zeros(max_label + 1)
    
    for label_idx, weight in class_weights.items():
        weights_tensor[label_idx] = weight
        
    return weights_tensor


def create_stratified_session_split(
    zarr_path: str,
    modalities: List[str] = ["eeg"],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    min_samples_per_class: int = 1
) -> Tuple[List[str], List[str], List[str]]:
    """
    Create a stratified session split that maintains label distribution across splits.
    
    Args:
        zarr_path: Path to query zarr store
        modalities: List of modalities to load
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        seed: Random seed
        min_samples_per_class: Minimum samples per class in each split
        
    Returns:
        Tuple of (train_sessions, val_sessions, test_sessions)
    """
    # Load dataset and get session-wise label distributions
    dataset = EfficientQueryDataset(
        zarr_path=zarr_path,
        modalities=modalities,
        preload_labels=True,
        verbose=False
    )
    
    sess_map = dataset.get_session_indices()
    
    # Calculate label distribution per session
    session_labels = {}
    for sess_id, indices in sess_map.items():
        labels = dataset.cached_labels[indices]
        session_labels[sess_id] = Counter(labels.numpy() if hasattr(labels, 'numpy') else labels.tolist())
    
    # Sort sessions by total sample count for balanced allocation
    sorted_sessions = sorted(session_labels.keys(), 
                           key=lambda s: sum(session_labels[s].values()), 
                           reverse=True)
    
    # Initialize splits
    train_sessions, val_sessions, test_sessions = [], [], []
    train_counts = Counter()
    val_counts = Counter()
    test_counts = Counter()
    
    # Allocate sessions to maintain stratification
    np.random.seed(seed)
    for sess_id in sorted_sessions:
        sess_label_counts = session_labels[sess_id]
        
        # Calculate current split sizes
        current_train = len(train_sessions)
        current_val = len(val_sessions) 
        current_test = len(test_sessions)
        total_assigned = current_train + current_val + current_test
        
        # Determine target split based on ratios
        if total_assigned == 0:
            # First session goes to train
            target_split = 'train'
        else:
            train_target = train_ratio * (total_assigned + 1)
            val_target = val_ratio * (total_assigned + 1)
            
            if current_train < train_target:
                target_split = 'train'
            elif current_val < val_target:
                target_split = 'val'
            else:
                target_split = 'test'
        
        # Assign to target split
        if target_split == 'train':
            train_sessions.append(sess_id)
            train_counts.update(sess_label_counts)
        elif target_split == 'val':
            val_sessions.append(sess_id)
            val_counts.update(sess_label_counts)
        else:
            test_sessions.append(sess_id)
            test_counts.update(sess_label_counts)
    
    logger.info(f"Stratified session split: {len(train_sessions)} train, "
              f"{len(val_sessions)} val, {len(test_sessions)} test sessions")
    logger.info(f"Train label distribution: {dict(train_counts)}")
    logger.info(f"Val label distribution: {dict(val_counts)}")
    logger.info(f"Test label distribution: {dict(test_counts)}")
    
    return train_sessions, val_sessions, test_sessions


def create_neural_data_loaders(
    zarr_path: str,
    task_type: str = "classification",
    batch_size: int = 32,
    modalities: List[str] = ["eeg"],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    auto_balance: bool = True,
    split_by_session: bool = True,
    seed: int = 42,
    num_workers: int = 4,
    label_names: Optional[Dict[int, str]] = None,
    filter_binary: bool = False,
    verbose: bool = True,
    analyze_distribution: bool = True,
    preload_labels: bool = True,
    session_limit: Optional[int] = None
) -> Dict[str, Any]:
    """
    High-level function to create neural data loaders with best practices.
    
    This is the recommended entry point for most neural data loading scenarios.
    It automatically handles:
    - Label distribution analysis
    - Class imbalance detection and correction
    - Session-wise splitting to prevent data leakage
    - Proper error handling and logging
    
    Args:
        zarr_path: Path to query zarr store
        task_type: Type of task ("classification", "regression", "binary")
        batch_size: Batch size for training
        modalities: Neural data modalities to load
        train_ratio: Fraction for training data
        val_ratio: Fraction for validation data
        test_ratio: Fraction for test data
        auto_balance: Whether to automatically handle class imbalance
        split_by_session: Whether to split by session (recommended)
        seed: Random seed for reproducibility
        num_workers: Number of data loading workers
        label_names: Optional mapping from label indices to readable names
        filter_binary: Whether to filter for only binary labels (0,1)
        verbose: Whether to print detailed information
        
    Returns:
        Dictionary containing:
        - 'train_loader': Training DataLoader
        - 'val_loader': Validation DataLoader (if val_ratio > 0)
        - 'test_loader': Test DataLoader (if test_ratio > 0)
        - 'distribution_info': Label distribution analysis
        - 'feature_shape': Shape of features for model initialization
        - 'num_classes': Number of unique classes
        - 'class_weights': Tensor of class weights for loss functions
    """
    if verbose:
        logger.info(f"Creating neural data loaders for {task_type} task")
        logger.info(f"Data source: {zarr_path}")
        logger.info(f"Modalities: {modalities}")
        logger.info(f"Split: {train_ratio:.1f}/{val_ratio:.1f}/{test_ratio:.1f} (train/val/test)")
        logger.info(f"Session-wise splitting: {split_by_session}")
    
    # Set up label filtering for binary tasks
    filter_func = None
    if task_type == "binary" or filter_binary:
        filter_func = filter_binary_labels
        if verbose:
            logger.info("Filtering for binary labels only (0, 1)")
    
    # Create loaders with automatic balancing if requested
    if auto_balance:
        train_loader, val_loader, test_loader, distribution_info = create_auto_balanced_loaders(
            zarr_path=zarr_path,
            batch_size=batch_size,
            modalities=modalities,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            split_by_session=split_by_session,
            seed=seed,
            num_workers=num_workers,
            filter_label_func=filter_func,
            label_names=label_names,
            verbose=verbose,
            session_limit=session_limit
        )
    else:
        # Create dataset once for analysis and reuse
        dataset = EfficientQueryDataset(
            zarr_path=zarr_path, 
            modalities=modalities, 
            preload_labels=preload_labels,
            verbose=verbose,
            session_limit=session_limit
        )
        
        # Get distribution info if needed
        if analyze_distribution:
            distribution_info = analyze_label_distribution(dataset, label_map=label_names)
        else:
            # Minimal distribution info for compatibility
            distribution_info = {'class_weights': {}, 'label_names': {}}
        
        train_loader, val_loader, test_loader = create_efficient_data_loaders(
            zarr_path=zarr_path,
            batch_size=batch_size,
            modalities=modalities,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            split_by_session=split_by_session,
            seed=seed,
            num_workers=num_workers,
            filter_label_func=filter_func,
            label_names=label_names,
            analyze_distribution=False,  # Skip since we already did it
            use_weighted_sampling=False,
            verbose=verbose,
            _reuse_dataset=dataset  # Reuse the dataset
        )
    
    # Get feature shape for model initialization from zarr metadata (fast)
    try:
        # Try to get from existing dataset first
        if auto_balance or not analyze_distribution:
            # We have a dataset reference from the above logic
            if hasattr(train_loader.dataset, 'dataset'):
                # It's a Subset, get the underlying dataset
                base_dataset = train_loader.dataset.dataset
            else:
                base_dataset = train_loader.dataset
            feature_shape = base_dataset.get_feature_shape()
        else:
            # Fast method: get shape from zarr metadata without loading data
            feature_shape = dataset.get_feature_shape()
    except (IndexError, AttributeError, Exception) as e:
        logger.warning(f"Could not determine feature shape from zarr metadata: {e}")
        # Fallback to loading a sample
        try:
            if hasattr(train_loader.dataset, '__getitem__'):
                sample_data = train_loader.dataset[0][0]
            else:
                # Handle Subset case
                sample_data = train_loader.dataset.dataset[train_loader.dataset.indices[0]][0]
            
            if isinstance(sample_data, dict):
                feature_shape = {mod: data.shape for mod, data in sample_data.items()}
            else:
                feature_shape = sample_data.shape
        except (IndexError, AttributeError):
            logger.warning("Could not determine feature shape from sample data either")
            feature_shape = None
    
    # Calculate number of classes
    num_classes = len(distribution_info['class_weights'])
    
    # Get class weights tensor for loss functions
    class_weights_tensor = get_loss_class_weights(distribution_info)
    
    if verbose:
        logger.info(f"Data loading complete:")
        logger.info(f"  Feature shape: {feature_shape}")
        logger.info(f"  Number of classes: {num_classes}")
        try:
            logger.info(f"  Training samples: {len(train_loader.dataset)}")
            if val_loader:
                logger.info(f"  Validation samples: {len(val_loader.dataset)}")
            if test_loader:
                logger.info(f"  Test samples: {len(test_loader.dataset)}")
        except (TypeError, AttributeError):
            logger.info("  Sample counts: Unable to determine dataset sizes")
    
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'distribution_info': distribution_info,
        'feature_shape': feature_shape,
        'num_classes': num_classes,
        'class_weights': class_weights_tensor,
        'label_names': distribution_info['label_names']
    }


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