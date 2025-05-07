"""
Functions for time-aligned windowing of EEG and fNIRS data.

This module creates windows from EEG and fNIRS data that respect the natural
frame boundaries and sampling rates of each modality, with proper timestamp alignment.
It handles cases where fNIRS frames may be skipped in the data.
"""

import numpy as np
import torch
import logging
from typing import Dict, List, Tuple, Optional, Union

# Set up logger
logger = logging.getLogger(__name__)


def create_windows(
    eeg_data: Optional[np.ndarray] = None,
    fnirs_data: Optional[np.ndarray] = None,
    eeg_timestamps: Optional[np.ndarray] = None,
    fnirs_timestamps: Optional[np.ndarray] = None,
    return_torch_tensors: bool = True
) -> List[Dict[str, Union[np.ndarray, torch.Tensor, Dict]]]:
    """
    Create time-aligned windows from EEG and fNIRS data, respecting frame boundaries.
    
    This function creates non-overlapping windows that align with natural frame boundaries:
    - Each window contains exactly 1 fNIRS frame (210ms)
    - Each window contains exactly 7 EEG frames (105 samples at 500Hz)
    
    The function handles:
    1. Time alignment - finding common start and end times between modalities
    2. Missing fNIRS frames - detecting gaps and using zero-filled placeholders
    3. Continuous EEG - maintaining all EEG frames without skipping
    
    Timestamps are expected to be in format where index 1 is the timestamp we want.
    
    Args:
        eeg_data: EEG data array (frames, channels, samples_per_frame)
        fnirs_data: fNIRS data array (frames, channels, 1)
        eeg_timestamps: Timestamps for EEG data
        fnirs_timestamps: Timestamps for fNIRS data
        return_torch_tensors: Whether to return PyTorch tensors (True) or NumPy arrays (False)
    
    Returns:
        List of dictionaries, each containing:
        - 'eeg': EEG data window
        - 'fnirs': fNIRS data window (zeros for missing frames)
        - 'metadata': Window metadata with 'missing_fnirs' flag if applicable
    """
    # Define fixed window parameters
    eeg_frames_per_window = 7  # 7 EEG frames per window
    fnirs_frames_per_window = 1  # 1 fNIRS frame per window
    window_size_ms = 210  # Fixed window size (210ms = 1 fNIRS frame)
    eeg_samples_per_frame = 15  # Assuming 15 samples per EEG frame
    
    # Define thresholds for detecting gaps
    fnirs_expected_interval_ms = 210  # Expected time between fNIRS frames
    fnirs_gap_threshold_ms = 300  # Threshold to detect gaps (>1.5x expected interval)
    
    # Check if both modalities are available
    has_eeg = eeg_data is not None and eeg_timestamps is not None and len(eeg_timestamps) > 0
    has_fnirs = fnirs_data is not None and fnirs_timestamps is not None and len(fnirs_timestamps) > 0
    
    # If neither modality is available, return empty list
    if not has_eeg and not has_fnirs:
        logger.warning("No EEG or fNIRS data available")
        return []
    
    # Find the common time range between both modalities
    start_time = 0
    end_time = float('inf')
    
    # Find earliest start time (latest of the two starts)
    if has_eeg:
        # Extract the second column (index 1) from timestamps
        eeg_start_time = float(eeg_timestamps[0][1])
        eeg_end_time = float(eeg_timestamps[-1][1])
        start_time = max(start_time, eeg_start_time)
        end_time = min(end_time, eeg_end_time)
    
    if has_fnirs:
        # Extract the second column (index 1) from timestamps
        fnirs_start_time = float(fnirs_timestamps[0][1])
        fnirs_end_time = float(fnirs_timestamps[-1][1])
        start_time = max(start_time, fnirs_start_time)
        end_time = min(end_time, fnirs_end_time)
    
    if has_eeg and has_fnirs:
        # Log the time range overlaps
        total_duration_ms = (end_time - start_time) * 1000
        eeg_duration_ms = (eeg_end_time - eeg_start_time) * 1000
        fnirs_duration_ms = (fnirs_end_time - fnirs_start_time) * 1000
        
        logger.info(f"Time ranges - EEG: {eeg_start_time:.3f} to {eeg_end_time:.3f} ({eeg_duration_ms:.1f}ms)")
        logger.info(f"Time ranges - fNIRS: {fnirs_start_time:.3f} to {fnirs_end_time:.3f} ({fnirs_duration_ms:.1f}ms)")
        logger.info(f"Common time range: {start_time:.3f} to {end_time:.3f} ({total_duration_ms:.1f}ms)")
        
        # Log percentage of total data being used
        eeg_percent = min(100, (total_duration_ms / eeg_duration_ms) * 100) if eeg_duration_ms > 0 else 0
        fnirs_percent = min(100, (total_duration_ms / fnirs_duration_ms) * 100) if fnirs_duration_ms > 0 else 0
        
        logger.info(f"Using {eeg_percent:.1f}% of EEG data, {fnirs_percent:.1f}% of fNIRS data")
        
        # Calculate how much data was trimmed from start and end
        eeg_start_trim_ms = (eeg_start_time - min(eeg_start_time, fnirs_start_time)) * 1000
        eeg_end_trim_ms = (max(eeg_end_time, fnirs_end_time) - eeg_end_time) * 1000
        fnirs_start_trim_ms = (fnirs_start_time - min(eeg_start_time, fnirs_start_time)) * 1000
        fnirs_end_trim_ms = (max(eeg_end_time, fnirs_end_time) - fnirs_end_time) * 1000
        
        # Calculate percentage of data trimmed for each modality
        eeg_trimmed_percent = ((eeg_start_trim_ms + eeg_end_trim_ms) / eeg_duration_ms) * 100 if eeg_duration_ms > 0 else 0
        fnirs_trimmed_percent = ((fnirs_start_trim_ms + fnirs_end_trim_ms) / fnirs_duration_ms) * 100 if fnirs_duration_ms > 0 else 0
        
        # Store the trim information for metadata
        trim_info = {
            'eeg_trimmed_start_ms': eeg_start_trim_ms,
            'eeg_trimmed_end_ms': eeg_end_trim_ms,
            'eeg_trimmed_percent': eeg_trimmed_percent,
            'fnirs_trimmed_start_ms': fnirs_start_trim_ms,
            'fnirs_trimmed_end_ms': fnirs_end_trim_ms,
            'fnirs_trimmed_percent': fnirs_trimmed_percent
        }
        
        # Warn if substantial amounts of data are being trimmed (>10% of original)
        trim_threshold_percent = 10.0  # Warning threshold percentage
        
        if eeg_start_trim_ms > 0 or eeg_end_trim_ms > 0:
            if eeg_trimmed_percent > trim_threshold_percent:
                logger.warning(f"Substantial EEG data trimmed ({eeg_trimmed_percent:.1f}%): " +
                               f"{eeg_start_trim_ms:.1f}ms from start, {eeg_end_trim_ms:.1f}ms from end")
        
        if fnirs_start_trim_ms > 0 or fnirs_end_trim_ms > 0:
            if fnirs_trimmed_percent > trim_threshold_percent:
                logger.warning(f"Substantial fNIRS data trimmed ({fnirs_trimmed_percent:.1f}%): " +
                               f"{fnirs_start_trim_ms:.1f}ms from start, {fnirs_end_trim_ms:.1f}ms from end")
    
    # Find the indices of frames within the valid time range
    eeg_start_idx = 0
    eeg_end_idx = 0
    fnirs_start_idx = 0
    fnirs_end_idx = 0
    
    if has_eeg:
        # Find first EEG timestamp >= start_time
        for i in range(len(eeg_timestamps)):
            if float(eeg_timestamps[i][1]) >= start_time:
                # Round down to the nearest frame boundary
                eeg_start_idx = (i // eeg_samples_per_frame)
                break
        
        # Find last EEG timestamp <= end_time
        for i in range(len(eeg_timestamps) - 1, -1, -1):
            if float(eeg_timestamps[i][1]) <= end_time:
                # Round up to next frame boundary to include the full frame
                eeg_end_idx = (i // eeg_samples_per_frame) + 1
                break
    
    if has_fnirs:
        # Find first fNIRS timestamp >= start_time
        for i in range(len(fnirs_timestamps)):
            if float(fnirs_timestamps[i][1]) >= start_time:
                fnirs_start_idx = i
                break
        
        # Find last fNIRS timestamp <= end_time
        for i in range(len(fnirs_timestamps) - 1, -1, -1):
            if float(fnirs_timestamps[i][1]) <= end_time:
                fnirs_end_idx = i + 1  # Include this frame
                break
    
    # Get dimensions after alignment
    n_eeg_frames = max(0, eeg_end_idx - eeg_start_idx) if has_eeg else 0
    n_fnirs_frames = max(0, fnirs_end_idx - fnirs_start_idx) if has_fnirs else 0
    
    logger.info(f"After alignment: EEG frames {eeg_start_idx}-{eeg_end_idx}, fNIRS frames {fnirs_start_idx}-{fnirs_end_idx}")
    logger.info(f"Using {n_eeg_frames} EEG frames, {n_fnirs_frames} fNIRS frames")
    
    # Create windows by going through all EEG frames within the common time range
    windows = []
    total_missing_fnirs = 0
    
    # Create fNIRS timestamp mapping (for fast lookups)
    fnirs_time_map = {}
    if has_fnirs:
        for i in range(fnirs_start_idx, fnirs_end_idx):
            # Get scalar timestamp and use as key
            ts = float(fnirs_timestamps[i][1])
            key = f"{ts:.6f}"
            fnirs_time_map[key] = i
    
    # Process frames based on EEG continuity (in the common time range)
    if has_eeg and n_eeg_frames >= eeg_frames_per_window:
        max_eeg_windows = n_eeg_frames // eeg_frames_per_window
        for window_idx in range(max_eeg_windows):
            # Calculate EEG frame indices for this window
            eeg_idx_start = eeg_start_idx + (window_idx * eeg_frames_per_window)
            eeg_idx_end = eeg_idx_start + eeg_frames_per_window
            
            # Skip incomplete windows or windows beyond our time range
            if eeg_idx_end > eeg_end_idx:
                continue
            
            # Extract EEG window data
            eeg_window = eeg_data[eeg_idx_start:eeg_idx_end].copy()
            
            # Calculate EEG window timestamps
            eeg_sample_start = eeg_idx_start * eeg_samples_per_frame
            eeg_sample_end = min(eeg_idx_end * eeg_samples_per_frame - 1, len(eeg_timestamps) - 1)
            eeg_start_time = float(eeg_timestamps[eeg_sample_start][1])
            eeg_end_time = float(eeg_timestamps[eeg_sample_end][1])
            
            # Calculate the expected fnirs timestamp for this window
            # We use the midpoint of the EEG window as the expected fNIRS time
            expected_fnirs_time = (eeg_start_time + eeg_end_time) / 2
            expected_key = f"{expected_fnirs_time:.6f}"
            
            # Try to find a nearby fNIRS frame
            missing_fnirs = True
            fnirs_idx = None
            
            # First, look for exact match
            if expected_key in fnirs_time_map:
                fnirs_idx = fnirs_time_map[expected_key]
                missing_fnirs = False
            else:
                # Look for nearest fNIRS timestamp within threshold
                nearest_time_diff = float('inf')
                for ts_key, idx in fnirs_time_map.items():
                    time_diff = abs(float(ts_key) - expected_fnirs_time)
                    if time_diff < nearest_time_diff and time_diff * 1000 < fnirs_gap_threshold_ms / 2:
                        nearest_time_diff = time_diff
                        fnirs_idx = idx
                
                if fnirs_idx is not None:
                    missing_fnirs = False
            
            # Create fNIRS window - either real data or zeros for missing frames
            if not missing_fnirs and has_fnirs:
                fnirs_window = fnirs_data[fnirs_idx:fnirs_idx+1].copy()
                fnirs_start_time = float(fnirs_timestamps[fnirs_idx][1])
                fnirs_end_time = fnirs_start_time  # Same as start time for single frame
            else:
                # Create zero-filled placeholder for missing fNIRS frame
                missing_fnirs = True
                total_missing_fnirs += 1
                
                # Use the shape of real fNIRS data if available
                if has_fnirs and fnirs_data.shape[1:]:
                    fnirs_window = np.zeros((1,) + fnirs_data.shape[1:], dtype=np.float32)
                else:
                    fnirs_window = np.zeros((1, 1, 1), dtype=np.float32)
                
                # Use EEG timestamps for missing fNIRS
                fnirs_start_time = expected_fnirs_time
                fnirs_end_time = expected_fnirs_time
            
            # Create window metadata
            metadata = {
                'window_idx': window_idx,
                'eeg_frame_range': (eeg_idx_start, eeg_idx_end),
                'fnirs_frame_range': (fnirs_idx, fnirs_idx + 1) if fnirs_idx is not None else None,
                'eeg_start_time': eeg_start_time,
                'eeg_end_time': eeg_end_time,
                'fnirs_start_time': fnirs_start_time,
                'fnirs_end_time': fnirs_end_time,
                'window_duration_ms': window_size_ms,
                'has_eeg': True,
                'has_fnirs': has_fnirs,
                'missing_fnirs': missing_fnirs,
                'alignment_start_time': start_time,
                'alignment_end_time': end_time
            }
            
            # Add trim information to the first window's metadata
            if window_idx == 0 and 'trim_info' in locals():
                metadata.update(trim_info)
            
            # Convert to PyTorch tensors if requested
            if return_torch_tensors:
                try:
                    eeg_window = torch.from_numpy(eeg_window)
                    fnirs_window = torch.from_numpy(fnirs_window)
                except (ImportError, NameError):
                    # If torch is not available, use numpy arrays
                    pass
            
            # Add to window list
            windows.append({
                'eeg': eeg_window,
                'fnirs': fnirs_window,
                'metadata': metadata
            })
    
    # Log summary
    if has_eeg and has_fnirs:
        missing_percentage = (total_missing_fnirs / len(windows)) * 100 if len(windows) > 0 else 0
        logger.info(f"Created {len(windows)} windows, {total_missing_fnirs} ({missing_percentage:.1f}%) with missing fNIRS data")
    else:
        logger.info(f"Created {len(windows)} windows using only {'EEG' if has_eeg else 'fNIRS'} data")
    
    # Add a final summary to metadata of last window if we have one
    if windows:
        last_window = windows[-1]
        last_window['metadata']['total_windows'] = len(windows)
        last_window['metadata']['total_missing_fnirs'] = total_missing_fnirs
        last_window['metadata']['missing_fnirs_percentage'] = (total_missing_fnirs / len(windows)) * 100 if len(windows) > 0 else 0
        last_window['metadata']['common_time_range_ms'] = (end_time - start_time) * 1000
    
    return windows