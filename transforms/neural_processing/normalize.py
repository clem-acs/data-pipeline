# data-pipeline/neural_processing/normalize.py
import numpy as np
import logging
import os
import json
from typing import Any, Iterable, List, Optional, Dict, Tuple

# Import the distance calculation function from the sibling module
try:
    from .fnirs_preprocessing import calculate_source_detector_distance
except ImportError:
    # Fallback for direct execution or different project structures
    try:
        from fnirs_preprocessing import calculate_source_detector_distance
    except ImportError:
        calculate_source_detector_distance = None
        logging.getLogger(__name__).warning(
            "Could not import calculate_source_detector_distance from fnirs_preprocessing."
        )

logger = logging.getLogger(__name__)

# Constants for fNIRS index decoding
NUM_DETECTOR_IDS_PER_MODULE = 6
NUM_SOURCE_IDS_PER_MODULE = 3
NUM_MODULES = 48
NUM_MOMENTS = 3
NUM_WAVELENGTHS = 2

def _decode_fnirs_index(original_channel_index: int) -> Optional[Dict[str, int]]:
    """
    Decode an fNIRS channel index into its component parts.
    
    Args:
        original_channel_index: The raw channel index to decode
        
    Returns:
        Dictionary with decoded components or None if decoding fails
    """
    try:
        detector_id_minus_1 = original_channel_index % NUM_DETECTOR_IDS_PER_MODULE
        temp = original_channel_index // NUM_DETECTOR_IDS_PER_MODULE
        detector_module_minus_1 = temp % NUM_MODULES
        temp = temp // NUM_MODULES
        source_id_minus_1 = temp % NUM_SOURCE_IDS_PER_MODULE
        temp = temp // NUM_MODULES
        source_module_minus_1 = temp % NUM_MODULES
        temp = temp // NUM_MODULES
        moment_idx = temp % NUM_MOMENTS
        wavelength_idx = temp // NUM_WAVELENGTHS

        if not (0 <= wavelength_idx < NUM_WAVELENGTHS and \
                0 <= moment_idx < NUM_MOMENTS and \
                0 <= source_module_minus_1 < NUM_MODULES and \
                0 <= source_id_minus_1 < NUM_SOURCE_IDS_PER_MODULE and \
                0 <= detector_module_minus_1 < NUM_MODULES and \
                0 <= detector_id_minus_1 < NUM_DETECTOR_IDS_PER_MODULE):
            logger.error(f"Decoded index component out of bounds for index {original_channel_index}.")
            return None
        return {
            'wavelength_idx': wavelength_idx,
            'moment_idx': moment_idx,
            'source_module': source_module_minus_1 + 1,
            'source_id': source_id_minus_1 + 1,
            'detector_module': detector_module_minus_1 + 1,
            'detector_id': detector_id_minus_1 + 1
        }
    except Exception as e:
        logger.error(f"Error decoding fNIRS index {original_channel_index}: {e}")
        return None

def normalize_window_dataset(
    window_dataset: Any,
    retained_fnirs_indices: Optional[List[int]] = None
) -> Any:
    """
    Initial exploratory function to understand fNIRS window data.
    
    This function:
    1. Examines the structure of window_dataset
    2. Extracts and aggregates fnirs data by moment and wavelength
    3. Reports basic statistics to help understand data distribution
    
    Args:
        window_dataset: Dataset containing windowed data
        retained_fnirs_indices: List of retained fNIRS channel indices (optional)
                               If not provided, will try to analyze window structure directly
        
    Returns:
        The unmodified window_dataset (placeholder for future normalization)
    """
    # Log information about received data
    if retained_fnirs_indices is not None:
        logger.warning(f"NORMALIZE: Received {len(retained_fnirs_indices)} retained fNIRS channel indices.")
        if len(retained_fnirs_indices) > 10:
            logger.info(f"NORMALIZE: Retained fNIRS indices (sample): {retained_fnirs_indices[:5]}...{retained_fnirs_indices[-5:]}")
        else:
            logger.info(f"NORMALIZE: Retained fNIRS indices: {retained_fnirs_indices}")
    else:
        logger.warning("NORMALIZE: No retained_fnirs_indices received. Will attempt to analyze window data directly.")

    if not window_dataset:
        logger.warning("NORMALIZE: Window dataset is None or empty. Skipping exploration.")
        return window_dataset

    # Try to determine window count
    num_windows = 0
    try:
        num_windows = len(window_dataset)
        if num_windows == 0:
            logger.warning("NORMALIZE: Window dataset is empty (length is 0). Skipping.")
            return window_dataset
        logger.warning(f"NORMALIZE: Processing window_dataset with {num_windows} windows.")
    except TypeError:
        logger.warning("NORMALIZE: Window dataset does not support len(). Processing as an iterable.")

    # Prepare data structures for aggregating statistics
    all_eeg_values = []
    processed_windows_count = 0
    
    # Track statistics by moment and wavelength
    moment_wavelength_stats = {
        moment: {
            wavelength: {
                'values': [],
                'count': 0,
            } for wavelength in range(NUM_WAVELENGTHS)
        } for moment in range(NUM_MOMENTS)
    }
    
    # Flag for shape mismatch logging
    fnirs_shape_mismatch_logged = False
    
    # If retained_fnirs_indices not provided, try to determine from first window
    if retained_fnirs_indices is None:
        # Try to examine the first window to get shape information
        try:
            if num_windows > 0:
                first_window = window_dataset[0]
                fnirs_data = None
                
                if isinstance(first_window, dict) and 'fnirs' in first_window:
                    fnirs_data = first_window['fnirs']
                elif hasattr(first_window, 'fnirs'):
                    fnirs_data = first_window.fnirs
                
                if fnirs_data is not None and isinstance(fnirs_data, np.ndarray):
                    # Get number of channels from first dimension
                    num_channels = fnirs_data.shape[0]
                    logger.warning(f"NORMALIZE: Detected {num_channels} fNIRS channels from first window")
                    
                    # Create sequential indices (0 to num_channels-1)
                    retained_fnirs_indices = list(range(num_channels))
                    logger.warning(f"NORMALIZE: Created synthetic channel indices (0 to {num_channels-1})")
        except Exception as e:
            logger.error(f"NORMALIZE: Failed to extract channel information from first window: {e}")
    
    # Initialize channel values dictionary
    fnirs_channel_values: Dict[int, List[float]] = {idx: [] for idx in retained_fnirs_indices} if retained_fnirs_indices else {}

    # Process each window
    for i, window in enumerate(window_dataset):
        processed_windows_count += 1
        if window is None: continue

        # Extract data from window
        eeg_data, fnirs_data_in_window = None, None
        if isinstance(window, dict):
            eeg_data = window.get('eeg')
            fnirs_data_in_window = window.get('fnirs')
        elif hasattr(window, 'eeg') or hasattr(window, 'fnirs'):
            eeg_data = getattr(window, 'eeg', None)
            fnirs_data_in_window = getattr(window, 'fnirs', None)

        # Process EEG data (simple collection for overall mean)
        if eeg_data is not None and isinstance(eeg_data, np.ndarray) and eeg_data.size > 0:
            all_eeg_values.append(eeg_data.flatten())

        # Process fNIRS data if available
        if fnirs_data_in_window is not None and isinstance(fnirs_data_in_window, np.ndarray) and \
           fnirs_data_in_window.size > 0 and retained_fnirs_indices:
            
            # Extract shape information
            num_channels_in_window_data = fnirs_data_in_window.shape[0]
            num_frames_in_segment = fnirs_data_in_window.shape[1] if fnirs_data_in_window.ndim > 1 else 1

            # Validate shape matches expectations
            if num_channels_in_window_data != len(retained_fnirs_indices):
                if not fnirs_shape_mismatch_logged:
                     logger.error(
                         f"NORMALIZE CRITICAL MISMATCH: fnirs_data in window has {num_channels_in_window_data} channels (shape[0]), "
                         f"but {len(retained_fnirs_indices)} retained_fnirs_indices were provided. "
                         f"Window data shape: {fnirs_data_in_window.shape}. "
                         f"This indicates a fundamental issue in how windowed fNIRS data maps to retained_fnirs_indices."
                     )
                     fnirs_shape_mismatch_logged = True
                continue 
            
            # Average each channel over frames in this window
            if num_frames_in_segment > 0:
                mean_channel_values_this_window = np.nanmean(fnirs_data_in_window, axis=1)
            else:
                mean_channel_values_this_window = np.full(num_channels_in_window_data, np.nan)

            # Process each channel's data
            if mean_channel_values_this_window.ndim == 1 and mean_channel_values_this_window.shape[0] == len(retained_fnirs_indices):
                for ch_idx_in_retained_list, original_global_idx in enumerate(retained_fnirs_indices):
                    value = mean_channel_values_this_window[ch_idx_in_retained_list]
                    if np.isfinite(value): 
                        fnirs_channel_values[original_global_idx].append(value)
                        
                        # Decode the index to get moment and wavelength
                        decoded = _decode_fnirs_index(original_global_idx)
                        if decoded:
                            moment = decoded['moment_idx']
                            wavelength = decoded['wavelength_idx']
                            moment_wavelength_stats[moment][wavelength]['values'].append(value)
                            moment_wavelength_stats[moment][wavelength]['count'] += 1
            else:
                if not fnirs_shape_mismatch_logged:
                     logger.warning(
                         f"NORMALIZE: Post-averaging shape mismatch for fNIRS data in window {i}. "
                         f"Expected 1D array of length {len(retained_fnirs_indices)}, "
                         f"got shape {mean_channel_values_this_window.shape}. "
                         f"Original window data shape: {fnirs_data_in_window.shape}."
                     )
                     fnirs_shape_mismatch_logged = True

    # Report summary statistics - basics first
    if not processed_windows_count:
        logger.warning("NORMALIZE: No windows were processed successfully.")
        return window_dataset
        
    logger.warning(f"NORMALIZE: Processed {processed_windows_count} windows successfully")

    # Report EEG statistics if available
    if all_eeg_values:
        concatenated_eeg = np.concatenate(all_eeg_values)
        if concatenated_eeg.size > 0:
            logger.warning(f"NORMALIZE EEG STATS: mean={np.nanmean(concatenated_eeg):.4f}, std={np.nanstd(concatenated_eeg):.4f}")
        else:
            logger.warning("NORMALIZE: No EEG data points found for statistics.")
    else:
        logger.warning("NORMALIZE: No EEG data arrays found.")

    # Report fNIRS statistics by moment and wavelength
    if fnirs_channel_values:
        logger.warning(f"============ NORMALIZE: fNIRS DATA STATISTICS ({len(fnirs_channel_values)} CHANNELS) ============")
        max_distance = 0
        all_distances = []
        
        # Get basic stats by moment and wavelength
        for moment in range(NUM_MOMENTS):
            for wavelength in range(NUM_WAVELENGTHS):
                stats = moment_wavelength_stats[moment][wavelength]
                if stats['values']:
                    mean_val = np.nanmean(stats['values'])
                    std_val = np.nanstd(stats['values'])
                    min_val = np.nanmin(stats['values'])
                    max_val = np.nanmax(stats['values'])
                    logger.warning(
                        f"NORMALIZE: Moment {moment}, Wavelength {wavelength}: "
                        f"mean={mean_val:.4e}, std={std_val:.4e}, "
                        f"min={min_val:.4e}, max={max_val:.4e}, "
                        f"count={stats['count']}"
                    )
                else:
                    logger.warning(f"NORMALIZE: Moment {moment}, Wavelength {wavelength}: No data")
        
        # Calculate distances if the calculation function is available
        if calculate_source_detector_distance is not None:
            # Try to load layout data
            layout_data = None
            current_script_dir = os.path.dirname(os.path.abspath(__file__))
            layout_json_path = os.path.join(current_script_dir, 'layout.json')
            
            if os.path.exists(layout_json_path):
                try:
                    with open(layout_json_path, 'r') as f:
                        layout_data = json.load(f)
                    logger.info(f"Successfully loaded layout data from {layout_json_path}")
                except Exception as e:
                    logger.error(f"Could not load layout data from {layout_json_path}: {e}")
                    layout_data = None
            
            if layout_data:
                # Compute distances for each channel
                for original_ch_idx in fnirs_channel_values.keys():
                    decoded = _decode_fnirs_index(original_ch_idx)
                    if not decoded:
                        continue
                    
                    try:
                        distance = calculate_source_detector_distance(
                            decoded['source_module'], decoded['source_id'],
                            decoded['detector_module'], decoded['detector_id'],
                            layout_data
                        )
                        if np.isfinite(distance):
                            all_distances.append(distance)
                            max_distance = max(max_distance, distance)
                    except Exception as e:
                        logger.warning(f"Error calculating distance for channel {original_ch_idx}: {e}")
                
                # Report distance statistics
                if all_distances:
                    logger.warning(f"NORMALIZE DISTANCE STATS: min={min(all_distances):.2f}mm, "
                               f"max={max_distance:.2f}mm, "
                               f"mean={np.mean(all_distances):.2f}mm, "
                               f"median={np.median(all_distances):.2f}mm")
                else:
                    logger.warning("NORMALIZE: No valid distances could be calculated.")
            else:
                logger.warning("NORMALIZE: Distance calculations skipped - no layout data available.")
        else:
            logger.warning("NORMALIZE: Distance calculations skipped - distance calculation function not available.")
    else:
        logger.warning("NORMALIZE: No fNIRS channel values collected.")

    logger.warning("NORMALIZE: Exploratory analysis complete. Returning window_dataset unmodified.")
    return window_dataset
