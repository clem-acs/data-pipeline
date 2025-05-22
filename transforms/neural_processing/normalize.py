import logging
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict

# --- Constants ---
NUM_MOMENTS = 3
NUM_WAVELENGTHS = 2
# Constants for fNIRS index decoding (restored based on original logic)
NUM_DETECTOR_IDS_PER_MODULE = 6
NUM_SOURCE_IDS_PER_MODULE = 3
NUM_MODULES = 48 # This was used for source_module_zb and detector_module_zb decoding

# --- Logger Setup ---
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO) # Default level

# --- Core Decoding Function (Restored Original Logic) ---
def _decode_fnirs_index(original_global_idx: int) -> Optional[Dict[str, int]]:
    """
    Decode an fNIRS channel index into its component parts.
    This version is restored based on the encoding from
    data-pipeline/transforms/neural_processing/fnirs_preprocessing.py:
    index = ((((wavelength_idx * NUM_MOMENTS + moment_idx) * NUM_MODULES +
                (source_module-1)) * NUM_SOURCE_IDS_PER_MODULE + (source_id-1)) * NUM_MODULES +
               (detector_module-1)) * NUM_DETECTOR_IDS_PER_MODULE + (detector_id-1)

    Args:
        original_global_idx: The raw channel index to decode

    Returns:
        Dictionary with decoded components or None if decoding fails
    """
    temp_val = original_global_idx

    # Detector ID (1-6, 0-5 zero-based)
    detector_id_zb = temp_val % NUM_DETECTOR_IDS_PER_MODULE
    temp_val //= NUM_DETECTOR_IDS_PER_MODULE

    # Detector Module (1-48, 0-47 zero-based)
    detector_module_zb = temp_val % NUM_MODULES
    temp_val //= NUM_MODULES

    # Source ID (1-3, 0-2 zero-based)
    source_id_zb = temp_val % NUM_SOURCE_IDS_PER_MODULE
    temp_val //= NUM_SOURCE_IDS_PER_MODULE

    # Source Module (1-48, 0-47 zero-based)
    source_module_zb = temp_val % NUM_MODULES
    temp_val //= NUM_MODULES

    # Moment Index (0-2)
    moment_idx = temp_val % NUM_MOMENTS
    temp_val //= NUM_MOMENTS

    # Wavelength Index (0-1)
    wavelength_idx = temp_val % NUM_WAVELENGTHS

    # Check if the index was too large / fully decoded
    # If temp_val // NUM_WAVELENGTHS is not 0, the original_global_idx was
    # larger than the maximum possible index derived from the constants.
    if temp_val // NUM_WAVELENGTHS != 0:
        # logger.warning( # Minimal logging for direct output
        #     f"Invalid fNIRS index {original_global_idx}: "
        #     f"exceeds maximum possible value based on current constants. "
        #     f"Residual quotient after decoding all components: {temp_val // NUM_WAVELENGTHS}."
        # )
        return None

    return {
        'wavelength_idx': wavelength_idx,
        'moment_idx': moment_idx,
        'source_module': source_module_zb + 1,  # Convert to 1-based
        'source_id': source_id_zb + 1,          # Convert to 1-based
        'detector_module': detector_module_zb + 1,  # Convert to 1-based
        'detector_id': detector_id_zb + 1,          # Convert to 1-based
    }

# --- Mask Creation Function ---
def create_moment_wavelength_masks(
    retained_fnirs_indices: List[int]
) -> Dict[Tuple[int, int], np.ndarray]:
    """
    Creates boolean masks for each moment/wavelength combination,
    aligned with the provided list of retained fNIRS indices.
    """
    if not retained_fnirs_indices:
        logger.info("retained_fnirs_indices is empty. Returning empty mask dictionary.")
        return {}

    num_retained_channels = len(retained_fnirs_indices)
    masks: Dict[Tuple[int, int], np.ndarray] = {}

    # Initialize a boolean mask (all False) for each combination
    for m_idx in range(NUM_MOMENTS):
        for w_idx in range(NUM_WAVELENGTHS):
            masks[(m_idx, w_idx)] = np.zeros(num_retained_channels, dtype=bool)

    # Populate the masks
    for i, original_global_idx in enumerate(retained_fnirs_indices):
        decoded_info = _decode_fnirs_index(original_global_idx)
        if decoded_info:
            moment_idx = decoded_info['moment_idx']
            wavelength_idx = decoded_info['wavelength_idx']
            combo_key = (moment_idx, wavelength_idx)
            if combo_key in masks: # Check if decoded combo is within expected range
                masks[combo_key][i] = True
            else:
                logger.warning(
                    f"Index {original_global_idx} decoded to M{moment_idx}, W{wavelength_idx}, "
                    f"which is outside the expected range defined by NUM_MOMENTS/NUM_WAVELENGTHS. "
                    f"This index will not be included in any mask."
                )
        else:
            logger.debug(f"Failed to decode fNIRS index: {original_global_idx}. It will be False in all masks.")

    return masks

# --- Main Normalization/Statistics Function ---
def normalize_window_dataset(
    window_dataset: List[Any],
    retained_fnirs_indices: List[int],
    retained_channel_validity_mask: np.ndarray
) -> Any:
    """
    Performs fNIRS data transformation (log for M0/M2) and then
    simplified data quality assessment and statistical aggregation,
    using moment/wavelength masks. Focuses on steps 4.1 and 4.2.
    """
    # Inputs are assumed to be valid and correctly typed as per the signature.
    # window_dataset is a non-empty list.
    # retained_fnirs_indices is a non-empty list of integers.
    # retained_channel_validity_mask is a NumPy boolean array
    #   with length matching len(retained_fnirs_indices).

    num_windows = len(window_dataset)
    logger.info(f"Processing window_dataset with {num_windows} windows.")
    
    # As per docstring, retained_fnirs_indices is non-empty.
    # Thus, len(retained_fnirs_indices) is safe and > 0.
    num_retained_channels = len(retained_fnirs_indices)
    moment_wavelength_masks = create_moment_wavelength_masks(retained_fnirs_indices)
    
    # --- Initialization for Step 4.1: Data Quality Percentages ---
    logger.info("Initializing counters for fNIRS data quality assessment (Step 4.1)...")
    total_data_points_by_combo: Dict[Tuple[int, int], int] = defaultdict(int)
    finite_nonzero_data_points_by_combo: Dict[Tuple[int, int], int] = defaultdict(int)

    # --- Initialization for Step 4.2: Detailed Statistics ---
    logger.info("Initializing data structures for detailed statistics (Step 4.2, post log transformation)...")
    fully_valid_moment_wavelength_data_points: Dict[Tuple[int, int], List[float]] = {
        (m, w): [] for m in range(NUM_MOMENTS) for w in range(NUM_WAVELENGTHS)
    }

    for window_data in window_dataset: # Iterate over the input parameter
        if window_data is None: continue
        window_metadata = window_data.get('metadata', {})
        is_fnirs_window_valid = window_metadata.get('fnirs_valid', True)

        if not is_fnirs_window_valid: continue

        raw_fnirs_data_this_window = window_data.get('fnirs')
        if raw_fnirs_data_this_window is None or not isinstance(raw_fnirs_data_this_window, np.ndarray):
            continue

        # --- Apply log transformation to a copy of fNIRS data for this window ---
        # Ensure data is float type to handle np.nan
        transformed_fnirs_data = raw_fnirs_data_this_window.astype(float, copy=True)
        
        # Suppress warnings for log(0) or log(negative), which return -inf/nan.
        # Then, explicitly set where original data was non-positive to np.nan.
        with np.errstate(divide='ignore', invalid='ignore'):
            transformed_fnirs_data = np.log(transformed_fnirs_data)
        transformed_fnirs_data[raw_fnirs_data_this_window <= 0] = np.nan
        # --- End log transformation ---

        if transformed_fnirs_data.shape[0] != num_retained_channels:
            logger.warning(f"Shape mismatch. Window fNIRS data (post-transform) has {transformed_fnirs_data.shape[0]} channels, expected {num_retained_channels}. Skipping window.")
            continue
        
        num_samples_per_channel_in_window = transformed_fnirs_data.shape[1] if transformed_fnirs_data.ndim == 2 else 1

        for (m_idx, w_idx), specific_combo_mask in moment_wavelength_masks.items():
            combo_key = (m_idx, w_idx)
            # Combine moment/wavelength mask with overall structural validity mask
            effective_channel_mask_for_combo = specific_combo_mask & retained_channel_validity_mask

            if not np.any(effective_channel_mask_for_combo): 
                # logger.debug(f"No structurally valid channels for combo {combo_key} in this window.")
                continue 

            # Select data for these channels from the current window (using transformed data)
            data_for_combo_this_window = transformed_fnirs_data[effective_channel_mask_for_combo, :]
            
            # --- STEP 4.1: Accumulate counts for Percentage of Finite, Non-Zero Data ---
            num_channels_for_this_combo_segment = np.sum(effective_channel_mask_for_combo)
            
            total_data_points_by_combo[combo_key] += num_channels_for_this_combo_segment * num_samples_per_channel_in_window
            
            # Count finite and non-zero values from the log-transformed data
            finite_values_in_segment = data_for_combo_this_window[np.isfinite(data_for_combo_this_window)]
            finite_nonzero_count_in_segment = np.sum(finite_values_in_segment != 0)
            finite_nonzero_data_points_by_combo[combo_key] += finite_nonzero_count_in_segment

            # --- STEP 4.2: Collect data for Detailed Statistics ---
            # Data for detailed stats should be finite (non-positive values became NaN during log transform)
            if finite_values_in_segment.size > 0:
                fully_valid_moment_wavelength_data_points[combo_key].extend(finite_values_in_segment.tolist())

    # --- After processing all windows, Calculate and Log Step 4.1 Percentages ---
    logger.info("--- fNIRS Data Quality: Percentage of Finite, Non-Zero Data Points (Post Log Transform) ---")
    for m_idx in range(NUM_MOMENTS):
        for w_idx in range(NUM_WAVELENGTHS):
            combo_key = (m_idx, w_idx)
            total_points = total_data_points_by_combo.get(combo_key, 0)
            finite_nonzero_points = finite_nonzero_data_points_by_combo.get(combo_key, 0)
            
            percentage = (finite_nonzero_points / total_points * 100) if total_points > 0 else 0
            logger.info(f"Moment {m_idx}, Wavelength {w_idx}:")
            logger.info(f"  Total data points considered: {total_points}")
            logger.info(f"  Finite, non-zero data points: {finite_nonzero_points}")
            logger.info(f"  Percentage of finite, non-zero data: {percentage:.2f}%")
            logger.info("-----------------------------------------------------")

    # --- Log Step 4.2 Detailed Statistics ---
    logger.info("--- Detailed fNIRS Statistics (Fully Valid Data, Post Log Transform) ---")
    for m_idx in range(NUM_MOMENTS):
        for w_idx in range(NUM_WAVELENGTHS):
            combo_key = (m_idx, w_idx)
            data_list = fully_valid_moment_wavelength_data_points.get(combo_key, [])

            # All fNIRS data is now log-transformed
            logger.info(f"--- Moment {m_idx}, Wavelength {w_idx} (log-transformed data) ---")

            if data_list:
                data_array = np.array(data_list)
                if data_array.size > 0:
                    min_val = np.min(data_array)
                    q1_val = np.percentile(data_array, 25)
                    mean_val = np.mean(data_array)
                    median_val = np.median(data_array)
                    q3_val = np.percentile(data_array, 75)
                    max_val = np.max(data_array)
                    std_val = np.std(data_array)
                    iqr_val = q3_val - q1_val
                    count_val = data_array.size

                    logger.info(f"  Count: {count_val}")
                    logger.info(f"  Min:   {min_val:.4e}")
                    logger.info(f"  Q1:    {q1_val:.4e}")
                    logger.info(f"  Mean:  {mean_val:.4e}")
                    logger.info(f"  Median:{median_val:.4e}")
                    logger.info(f"  Q3:    {q3_val:.4e}")
                    logger.info(f"  Max:   {max_val:.4e}")
                    logger.info(f"  Std:   {std_val:.4e}")
                    logger.info(f"  IQR:   {iqr_val:.4e}")
                else:
                    logger.info("  No finite data points found for this combination after all filters.")
            else:
                logger.info("  No data points collected for this combination.")
            logger.info("--------------------------------------------------------------------")

    logger.info("Simplified fNIRS normalization complete (currently just log transform).")

    return window_dataset
