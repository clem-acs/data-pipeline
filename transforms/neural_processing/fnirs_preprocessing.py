"""
Functions for preprocessing fNIRS data before windowing.

Terminology:
- VALID: Refers to data points that have finite values (not NaN or Inf). A data point is either valid or invalid.
- FEASIBLE: Refers to channels that meet certain criteria (module presence, distance constraints).
  A channel is feasible if it can theoretically produce good data based on spatial considerations.

The workflow:
1. Generate feasible channel indices based on which modules are present and distance constraints
2. Filter the fNIRS data to keep only the feasible channels
"""

import numpy as np
from scipy import signal
import logging

__all__ = ['get_module_channel_indices', 'preprocess_fnirs']

# Set up logger
logger = logging.getLogger("fnirs_preprocessing")


def calculate_source_detector_distance(source_module, source_id, detector_module, detector_id, layout_data):
    """
    Calculate the Euclidean distance between a source and detector position.

    Args:
        source_module: Module number for the source (1-48)
        source_id: ID of the source within the module (1-3)
        detector_module: Module number for the detector (1-48)
        detector_id: ID of the detector within the module (1-6)
        layout_data: Dictionary containing source and detector positions

    Returns:
        float: Euclidean distance in millimeters
    """
    # Get source position
    try:
        source_pos = layout_data["source_locations"][source_module-1][source_id-1]
    except IndexError:
        return float('inf')  # Return infinity if source doesn't exist

    # Get detector position
    try:
        detector_pos = layout_data["detector_locations"][detector_module-1][detector_id-1]
    except IndexError:
        return float('inf')  # Return infinity if detector doesn't exist

    # Calculate Euclidean distance
    source_pos = np.array(source_pos)
    detector_pos = np.array(detector_pos)
    distance = np.sqrt(np.sum((source_pos - detector_pos)**2))

    return distance


def get_feasible_channel_indices(present_modules, layout_data, max_distance_mm=60):
    """
    Generate channel indices for all present modules and track distance statistics.

    Args:
        present_modules: List of module numbers that are present (1-48)
        layout_data: Dictionary containing source and detector positions
        max_distance_mm: Recommended maximum distance threshold in millimeters (used for statistics only)

    Returns:
        List of all channel indices for present modules, along with distance statistics
    """
    # Generate feasible indices
    feasible_indices = []
    present_channels_exceeding_distance_threshold = 0

    print(f"Generating channel indices for {len(present_modules)} modules")

    for wavelength_idx in range(2):  # 0=Red, 1=IR
        for moment_idx in range(3):  # 0=Zeroth, 1=First, 2=Second
            for source_module in present_modules:
                for source_id in range(1, 4):  # 1-3
                    for detector_module in present_modules:
                        for detector_id in range(1, 7):  # 1-6
                            # Calculate the index for this channel
                            index = ((((wavelength_idx * 3 + moment_idx) * 48 +
                                    (source_module-1)) * 3 + (source_id-1)) * 48 +
                                    (detector_module-1)) * 6 + (detector_id-1)

                            # Step 2: Check distance constraint
                            distance = calculate_source_detector_distance(
                                source_module, source_id, detector_module, detector_id, layout_data)

                            # Step 3: Check if distance is within threshold
                            if distance <= max_distance_mm:
                                feasible_indices.append(index)
                            else:
                                present_channels_exceeding_distance_threshold += 1

    print(f"Found {len(feasible_indices)} channels for selected modules, {present_channels_exceeding_distance_threshold} channels exceed the recommended distance threshold")

    return sorted(feasible_indices)


def preprocess_fnirs(fnirs_data, metadata, layout_data=None, present_modules=None, max_distance_mm=60):
    """
    Preprocess fNIRS data before windowing.

    Args:
        fnirs_data: fNIRS data array of shape (num_frames, num_channels, 1)
        metadata: Dictionary of metadata about the signal
        layout_data: Dictionary containing source and detector positions
            If None, no filtering will be applied
        present_modules: List of module numbers present in the data
            If None, no filtering will be applied
        max_distance_mm: Maximum distance in millimeters between source and detector
            Default is 60mm

    Returns:
        Tuple of (preprocessed_fnirs_data, preprocessing_metadata)
    """
    # Initialize preprocessing metadata
    preprocessing_metadata = {}

    # If module information is provided, get channel indices
    if layout_data is not None and present_modules is not None:
        # Get all feasible channel indices for present modules and track distance statistics
        feasible_indices = get_feasible_channel_indices(present_modules, layout_data, max_distance_mm)

        # Add filtering info to metadata
        preprocessing_metadata.update({
            'distance_threshold_mm': max_distance_mm,
            'module_count': len(present_modules),
            'modules': present_modules.copy(),
            'total_channel_count': len(feasible_indices),
        })

        # Ensure indices are within the data range
        if len(fnirs_data.shape) == 3:
            max_index = fnirs_data.shape[1] - 1
            indices_in_range = [idx for idx in feasible_indices if idx <= max_index]

            if len(indices_in_range) != len(feasible_indices):
                print(f"Warning: {len(feasible_indices) - len(indices_in_range)} channels exceed data dimensions")
                preprocessing_metadata['out_of_range_channel_count'] = len(feasible_indices) - len(indices_in_range)

            preprocessing_metadata['used_channel_count'] = len(indices_in_range)
            
            # Check validity of all channels efficiently in one pass
            all_indices = set(range(min(fnirs_data.shape[1], max_index + 1)))
            included_indices = set(indices_in_range)
            excluded_indices = list(all_indices - included_indices)
            
            # Process all channels in batches to be memory-efficient
            try:
                # Only check a subset of frames for large datasets to save memory
                frame_limit = min(1000, fnirs_data.shape[0]) if fnirs_data.shape[0] > 1000 else fnirs_data.shape[0]
                
                # Track statistics
                included_valid_channels = 0  # Included channels with at least one valid value
                included_all_inf_channels = 0  # Included channels that are entirely -inf/NaN
                excluded_valid_channels = 0  # Excluded channels with at least one valid value
                
                # Process all channel indices in batches
                batch_size = 5000  # Process 5000 channels at a time
                
                # First check included channels
                if indices_in_range:
                    included_indices_list = list(included_indices)
                    for i in range(0, len(included_indices_list), batch_size):
                        batch_indices = included_indices_list[i:i+batch_size]
                        # Check for finite values in this batch
                        has_finite_values = np.isfinite(fnirs_data[:frame_limit, batch_indices, :]).any(axis=0)
                        included_valid_channels += np.sum(has_finite_values)
                        included_all_inf_channels += len(batch_indices) - np.sum(has_finite_values)
                
                # Then check excluded channels
                if excluded_indices:
                    for i in range(0, len(excluded_indices), batch_size):
                        batch_indices = excluded_indices[i:i+batch_size]
                        # Check for finite values in this batch
                        has_finite_values = np.isfinite(fnirs_data[:frame_limit, batch_indices, :]).any(axis=0)
                        excluded_valid_channels += np.sum(has_finite_values)
                        
                        # Log progress for large operations
                        if len(excluded_indices) > 10000 and i % 10000 == 0:
                            print(f"Checked {i}/{len(excluded_indices)} excluded channels for valid data...")
                
                # Add all statistics to metadata
                preprocessing_metadata['included_channels_with_valid_data'] = int(included_valid_channels)
                preprocessing_metadata['included_channels_all_invalid'] = int(included_all_inf_channels)
                preprocessing_metadata['excluded_channels_with_valid_data'] = int(excluded_valid_channels)
                
                # Calculate percentage of valid included channels
                if len(indices_in_range) > 0:
                    valid_channel_percentage = (included_valid_channels / len(indices_in_range)) * 100
                    preprocessing_metadata['valid_channel_percentage'] = float(valid_channel_percentage)
                
                # Log findings for included channels
                logger.info(f"{included_valid_channels} out of {len(indices_in_range)} included channels ({valid_channel_percentage:.1f}%) contain valid data")
                
                if included_all_inf_channels > 0:
                    logger.warning(f"{included_all_inf_channels} included channels contain only invalid data (NaN/Infinity values)")
                
                # Log findings for excluded channels
                if excluded_valid_channels > 0:
                    logger.warning(f"{excluded_valid_channels} channels with valid data were excluded based on distance constraints")
                    preprocessing_metadata['channels_with_valid_data_excluded'] = int(excluded_valid_channels)
                else:
                    logger.debug(f"All {len(excluded_indices)} excluded channels contained only invalid data (NaN/Infinity values)")
                    preprocessing_metadata['all_excluded_channels_invalid'] = True
                    
            except Exception as e:
                print(f"Error checking channel validity: {e}")
            
            # Only filter to keep channels within data range
            if indices_in_range:
                if len(indices_in_range) == fnirs_data.shape[1]:
                    # No need to filter if all channels are used
                    return fnirs_data, preprocessing_metadata
                else:
                    # Filter to keep only in-range channels
                    return fnirs_data[:, indices_in_range, :], preprocessing_metadata
            else:
                # No channels within range
                print("Warning: No channels within data range")
                preprocessing_metadata['warning'] = "No channels within data range"
                return fnirs_data, preprocessing_metadata
        else:
            # Unexpected data shape
            preprocessing_metadata['warning'] = f"Unexpected data shape {fnirs_data.shape}, expected (frames, channels, 1)"
            return fnirs_data, preprocessing_metadata

    # No filtering applied, return original data
    return fnirs_data, preprocessing_metadata
