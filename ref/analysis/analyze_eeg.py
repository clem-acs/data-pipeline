#!/usr/bin/env python3
"""
EEG Channel Standard Deviation Analysis

This script loads EEG data from HDF5 files and computes the standard deviation
for each channel, displaying the results as a bar chart.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import re
from scipy import stats
from scipy import signal
from sklearn.feature_selection import mutual_info_regression
import warnings

# Try to import FOOOF - if not available, we'll handle it gracefully
try:
    from fooof import FOOOF
    from fooof.bands import Bands
    from fooof.analysis import get_band_peak_fg
    FOOOF_AVAILABLE = True
except ImportError:
    FOOOF_AVAILABLE = False
    warnings.warn("FOOOF package not found. FOOOF analysis will not be available. Install with: pip install fooof")

def parse_h5list_file(h5list_path):
    """
    Parse the h5list.txt file to get information about H5 files.

    Args:
        h5list_path: Path to h5list.txt file

    Returns:
        Dictionary mapping file names to shapes
    """
    file_info = {}
    current_file = None

    with open(h5list_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        # Check for file header
        file_match = re.search(r'## File: (.+\.h5)', line)
        if file_match:
            current_file = file_match.group(1)
            file_info[current_file] = {}
            continue

        # Check for EEG shape
        shape_match = re.search(r'- eeg/frames: \((.+)\)', line)
        if shape_match and current_file:
            shape_str = shape_match.group(1)
            shape_tuple = tuple(map(int, shape_str.split(', ')))
            file_info[current_file]['eeg_shape'] = shape_tuple

    return file_info

def find_h5_files(data_dir):
    """
    Find H5 files in the data directory.

    Args:
        data_dir: Directory to search

    Returns:
        List of H5 file paths
    """
    h5_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".h5"):
                h5_files.append(os.path.join(root, file))

    return h5_files


def load_eeg_data(h5_file_path, transpose=False, exclude_channels=None):
    """
    Load EEG data from an H5 file.

    Args:
        h5_file_path: Path to the H5 file
        transpose: Whether to transpose dimensions 1 and 2 (samples and channels) before concatenation
        exclude_channels: List of channel indices to exclude from the data

    Returns:
        Numpy array of EEG data with shape [channels, frames]
    """
    print(f"Loading file: {os.path.basename(h5_file_path)}")

    # Open the H5 file
    with h5py.File(h5_file_path, 'r') as f:
        # Check if EEG device data exists
        if 'devices' in f and 'eeg' in f['devices']:
            eeg_group = f['devices']['eeg']

            # Get frames data
            if 'frames_data' in eeg_group:
                frames_data = eeg_group['frames_data'][:]
                print(f"EEG data shape: {frames_data.shape}")

                # Original format is [chunks, channels, samples]
                if transpose:
                    if len(frames_data.shape) >= 3:
                        frames_data = frames_data.transpose(0, 2, 1)  # Now
                    else:
                        print(f"WARNING: Cannot transpose data with shape {frames_data.shape}")
                        print(f"  - Need at least 3 dimensions to transpose dimensions 1 and 2")

                # Now reshape data to [channels, total_samples]
                if len(frames_data.shape) >= 3:
                        # Data is [chunks, channels, samples]
                        chunks = frames_data.shape[0]
                        num_channels = frames_data.shape[1]
                        samples_per_chunk = frames_data.shape[2]

                        # Reshape to put all samples in one dimension
                        total_samples = chunks * samples_per_chunk

                        # Create result array and fill it
                        # For standard data [chunks, channels, samples], we want [channels, chunks*samples]
                        # Instead of concatenating samples for each channel, we'll reshape the entire array directly

                        # First, permute dimensions to get [channels, chunks, samples]
                        rearranged = np.transpose(frames_data, (1, 0, 2))

                        # Then reshape each channel's data to flatten chunks and samples dimensions
                        reshaped_data = rearranged.reshape(num_channels, -1)

                        # Handle excluded channels if any
                        if exclude_channels:
                            # Convert string indices to integers if needed
                            if isinstance(exclude_channels, str):
                                exclude_channels = [int(ch.strip()) for ch in exclude_channels.split(',')]
                            elif not isinstance(exclude_channels, (list, tuple, np.ndarray)):
                                exclude_channels = [int(exclude_channels)]
                            
                            # Validate channel indices
                            valid_channels = [ch for ch in exclude_channels if 0 <= ch < num_channels]
                            invalid_channels = [ch for ch in exclude_channels if ch < 0 or ch >= num_channels]
                            
                            if invalid_channels:
                                print(f"Warning: Ignoring invalid channel indices: {invalid_channels}")
                            
                            if valid_channels:
                                print(f"Excluding channels: {valid_channels}")
                                
                                # Create mask of channels to keep
                                keep_mask = np.ones(num_channels, dtype=bool)
                                keep_mask[valid_channels] = False
                                keep_indices = np.where(keep_mask)[0]
                                
                                # Select only the channels to keep
                                reshaped_data = reshaped_data[keep_indices]
                                print(f"Data shape after channel exclusion: {reshaped_data.shape}")
                        
                        print(f"Reshape via direct array manipulation: {rearranged.shape} → {reshaped_data.shape}")
                else:
                    print(f"WARNING: Unexpected data shape {frames_data.shape}")
                    print(f"  - Expected at least 3 dimensions")
                    reshaped_data = frames_data  # Just return the data as-is

                print(f"Reshaped data: {reshaped_data.shape}")
                return reshaped_data
            else:
                print("No frames_data found in EEG group")
                return None
        else:
            print("No EEG device data found in the file")
            return None

def plot_segment_rejection(segment_ranges, keep_indices, segment_duration, output_path="segment_rejection.png", segment_times=None):
    """
    Create a plot showing which segments were kept and which were rejected.

    Args:
        segment_ranges: Array of range values for each segment
        keep_indices: Indices of segments that were kept
        segment_duration: Duration of each segment in seconds
        output_path: Path to save the plot
        segment_times: Optional array of segment center times in seconds for x-axis
    """
    try:
        plt.figure(figsize=(12, 6))

        # Create bar chart showing all segments, coloring kept vs rejected
        num_segments = len(segment_ranges)

        # Use provided segment times if available, otherwise create evenly spaced times
        if segment_times is None:
            segment_times = np.arange(num_segments) * segment_duration

        # Convert indices to numpy array for easier checking
        keep_indices_set = set(keep_indices)

        # Create colors and labels for bars
        colors = []
        for i in range(num_segments):
            if i in keep_indices_set:
                colors.append('steelblue')  # Kept segments
            else:
                colors.append('firebrick')  # Rejected segments

        # Create a bar chart with width based on segment spacing or fixed proportion for variable spacing
        if len(segment_times) > 1:
            avg_spacing = (segment_times[-1] - segment_times[0]) / (len(segment_times) - 1)
            bar_width = min(segment_duration, avg_spacing) * 0.8
        else:
            bar_width = segment_duration * 0.8

        plt.bar(segment_times, segment_ranges, color=colors, width=bar_width)

        # Add a threshold line if we can calculate the threshold
        if keep_indices:
            # The threshold is the highest value that was kept
            keep_values = [segment_ranges[i] for i in keep_indices]
            if keep_values:
                threshold = max(keep_values)
                plt.axhline(y=threshold, color='k', linestyle='--')
                plt.text(segment_times[-1]*0.9, threshold*1.05, 'Rejection Threshold',
                        ha='right', va='bottom')

        # Count kept and rejected segments
        kept_count = len(keep_indices)
        rejected_count = num_segments - kept_count

        # Add labels and title
        plt.xlabel('Segment Start Time (seconds)')
        plt.ylabel('Median Range Value')
        plt.title(f'Segment Rejection Analysis ({kept_count} kept, {rejected_count} rejected)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Add a legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='steelblue', label=f'Kept Segments ({kept_count})'),
            Patch(facecolor='firebrick', label=f'Rejected Segments ({rejected_count})')
        ]
        plt.legend(handles=legend_elements, loc='upper right')

        # Add percentage rejected in text box
        reject_percent = (rejected_count / num_segments) * 100
        plt.figtext(0.02, 0.02, f"Rejected: {reject_percent:.1f}% of segments",
                   fontsize=9, bbox=dict(facecolor='white', alpha=0.8))

        plt.tight_layout()
        plt.savefig(output_path)
        print(f"Segment rejection plot saved to {output_path}")
        plt.close()
    except Exception as e:
        print(f"Warning: Failed to create segment rejection plot: {e}")
        # Continue with processing even if plotting fails

def preprocess_log_transform(eeg_data):
    """
    Apply logarithmic transformation to EEG amplitude values.
    This helps to normalize highly skewed amplitude distributions.

    Args:
        eeg_data: EEG data with shape [channels, frames]

    Returns:
        Log-transformed EEG data
    """
    if eeg_data is None:
        return None

    # Create a copy to avoid modifying the original data
    transformed_data = eeg_data.copy()

    # Handle zero or negative values by adding a small offset to all values
    # Find minimum value across all channels
    min_value = np.min(transformed_data)

    # If we have zero or negative values, add an offset to make all values positive
    if min_value <= 0:
        offset = abs(min_value) + 1e-6  # Small epsilon to avoid log(0)
        transformed_data = transformed_data + offset
        print(f"Log transform: Added offset of {offset:.6f} to make all values positive")

    # Apply natural logarithm to all values
    transformed_data = np.log(transformed_data)

    print(f"Applied logarithmic transformation to amplitude values")
    print(f"  - Original range: [{np.min(eeg_data):.4f}, {np.max(eeg_data):.4f}]")
    print(f"  - Transformed range: [{np.min(transformed_data):.4f}, {np.max(transformed_data):.4f}]")

    return transformed_data

def preprocess_reject_segments(eeg_data, segment_duration=5.0, overlap=0.5, sample_rate=500, reject_ratio=0.5,
                              plot_rejection=True, plot_path="segment_rejection.png"):
    """
    Preprocess EEG data by dividing into overlapping time segments and rejecting segments with high amplitude ranges.

    Args:
        eeg_data: EEG data with shape [channels, frames]
        segment_duration: Duration of each segment in seconds (default: 5.0s)
        overlap: Overlap between segments in seconds on each side (default: 0.5s)
        sample_rate: Sampling rate in Hz (default: 500Hz)
        reject_ratio: Ratio of segments to reject based on highest range values (default: 0.5)
        plot_rejection: Whether to create a visualization of rejected segments (default: True)
        plot_path: Path to save the rejection visualization plot

    Returns:
        Preprocessed EEG data with high-range segments removed
    """
    if eeg_data is None or eeg_data.shape[1] < sample_rate:
        print("Not enough data for segment rejection preprocessing")
        return eeg_data

    num_channels = eeg_data.shape[0]
    total_samples = eeg_data.shape[1]

    # Calculate segment size and step size in samples
    segment_size = int(segment_duration * sample_rate)
    step_size = int((segment_duration - 2 * overlap) * sample_rate)  # Step size with overlap on both sides

    if step_size <= 0:
        print(f"Warning: Overlap too large ({overlap}s) for segment duration ({segment_duration}s)")
        step_size = segment_size // 4  # Default to 1/4 segment size as step if overlap is too large

    # Calculate number of segments with overlap
    num_segments = (total_samples - segment_size) // step_size + 1

    if num_segments < 2:
        print(f"Warning: Only {num_segments} segments available, need at least 2 for rejection preprocessing")
        return eeg_data

    # Initialize arrays to store segment information
    segment_ranges = np.zeros(num_segments)
    segment_indices = []
    segment_centers = []  # Store the center index of each segment for reconstruction

    # Calculate range for each segment across all channels
    for i in range(num_segments):
        start_idx = i * step_size
        end_idx = start_idx + segment_size
        center_idx = start_idx + segment_size // 2

        # Extract segment data
        segment_data = eeg_data[:, start_idx:end_idx]

        # Calculate range for each channel
        channel_ranges = calculate_channel_ranges(segment_data)

        # Take median of ranges across channels
        median_range = np.median(channel_ranges)

        # Store range, indices, and center point
        segment_ranges[i] = median_range
        segment_indices.append((start_idx, end_idx))
        segment_centers.append(center_idx)

    # Determine threshold for rejection based on reject_ratio
    num_segments_to_keep = int(num_segments * (1 - reject_ratio))
    if num_segments_to_keep < 1:
        num_segments_to_keep = 1  # Always keep at least one segment

    # Get indices of segments to keep (those with lower ranges)
    keep_segment_indices = np.argsort(segment_ranges)[:num_segments_to_keep]
    keep_indices_list = list(keep_segment_indices)  # Convert to list for plot function
    keep_segment_indices.sort()  # Sort to maintain temporal order

    # Create a map of which samples to keep
    keep_mask = np.zeros(total_samples, dtype=bool)

    # Print some stats about the ranges to help diagnose issues
    print(f"Segment median ranges - min: {np.min(segment_ranges):.4f}, max: {np.max(segment_ranges):.4f}, median: {np.median(segment_ranges):.4f}")

    # Mark samples from kept segments as True in the mask
    reject_threshold = np.sort(segment_ranges)[num_segments_to_keep-1] if num_segments_to_keep > 0 else float('inf')
    print(f"Rejection threshold: {reject_threshold:.4f}")

    # Ensure we're actually rejecting segments by checking if all segment ranges are identical
    if np.max(segment_ranges) == np.min(segment_ranges):
        print("WARNING: All segments have identical range values. No effective rejection will occur.")
        # In this case, randomly select segments to keep
        keep_segment_indices = np.random.choice(np.arange(num_segments), num_segments_to_keep, replace=False)
        keep_indices_list = list(keep_segment_indices)
        keep_segment_indices.sort()

    segments_kept = 0
    total_kept_points = 0
    for segment_idx in keep_segment_indices:
        start_idx, end_idx = segment_indices[segment_idx]
        keep_mask[start_idx:end_idx] = True
        segments_kept += 1
        total_kept_points += (end_idx - start_idx)

    # Calculate how many points are actually getting masked out
    points_rejected = total_samples - np.sum(keep_mask)

    print(f"Segments kept: {segments_kept}/{num_segments} ({segments_kept/num_segments*100:.1f}%)")
    print(f"Rejected points: {points_rejected}/{total_samples} ({points_rejected/total_samples*100:.1f}%)")

    # Extract only the kept samples from the original data
    # Count consecutive ranges of True values to assemble the clean data
    ranges_to_keep = []
    in_range = False
    range_start = 0

    for i in range(total_samples):
        if keep_mask[i] and not in_range:
            # Start of a new range
            range_start = i
            in_range = True
        elif not keep_mask[i] and in_range:
            # End of a range
            ranges_to_keep.append((range_start, i))
            in_range = False

    # Add the last range if we're still in one
    if in_range:
        ranges_to_keep.append((range_start, total_samples))

    # Calculate total kept samples
    total_kept_samples = sum(end - start for start, end in ranges_to_keep)

    # If no data was actually rejected or too little data left, return original
    if points_rejected == 0:
        print("WARNING: No points were rejected. Returning original data.")
        return eeg_data

    if total_kept_samples < sample_rate * 2:  # Less than 2 seconds of data left
        print(f"WARNING: Too few samples left after rejection ({total_kept_samples}). Returning original data.")
        return eeg_data

    # Create new array for preprocessed data
    preprocessed_data = np.zeros((num_channels, total_kept_samples))

    # Fill in the preprocessed data array with kept ranges
    current_idx = 0
    for start, end in ranges_to_keep:
        range_length = end - start
        preprocessed_data[:, current_idx:current_idx + range_length] = eeg_data[:, start:end]
        current_idx += range_length

    # Final stats about the preprocessing
    print(f"Preprocessing summary:")
    print(f"  - Rejected {num_segments - num_segments_to_keep} of {num_segments} segments")
    print(f"  - Original data: {eeg_data.shape[1]} samples ({eeg_data.shape[1]/sample_rate:.1f}s)")
    print(f"  - Preprocessed data: {preprocessed_data.shape[1]} samples ({preprocessed_data.shape[1]/sample_rate:.1f}s)")
    print(f"  - Retained {total_kept_samples / total_samples * 100:.1f}% of original samples")

    # Create visualization of rejected segments if requested
    if plot_rejection:
        # Convert segment centers to seconds for x-axis
        segment_times_seconds = [center / sample_rate for center in segment_centers]
        plot_segment_rejection(segment_ranges, keep_indices_list, segment_duration, plot_path, segment_times_seconds)

    return preprocessed_data

def analyze_channel_std(eeg_data, num_eeg_channels=22):
    """
    Calculate standard deviation for each EEG channel.

    Args:
        eeg_data: EEG data with shape [channels, frames]
        num_eeg_channels: Number of EEG channels (excluding auxiliary channels)

    Returns:
        Array of standard deviations per channel
    """
    if eeg_data is None:
        return None

    # Use only EEG channels (e.g., first 22 for gTec)
    eeg_data = eeg_data[:num_eeg_channels, :]

    # Calculate standard deviation for each channel
    std_per_channel = np.std(eeg_data, axis=1)

    return std_per_channel

def plot_channel_std(std_per_channel, output_path="plot.png"):
    """
    Create a bar chart of standard deviations per channel.

    Args:
        std_per_channel: Array of standard deviations
        output_path: Path to save the plot
    """
    if std_per_channel is None:
        print("No data to plot")
        return

    plt.figure(figsize=(12, 6))
    channel_indices = np.arange(len(std_per_channel))

    plt.bar(channel_indices, std_per_channel, color='steelblue')
    plt.xlabel('Channel Index')
    plt.ylabel('Standard Deviation')
    plt.title('EEG Channel Standard Deviation')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(channel_indices)

    # Add values on top of bars
    for i, std in enumerate(std_per_channel):
        plt.text(i, std + 0.1, f'{std:.2f}', ha='center')

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")
    plt.close()

def normalize_channel(data):
    """
    Normalize a channel's data to have zero mean and unit variance (z-score).

    Args:
        data: Channel data array

    Returns:
        Normalized data array
    """
    # Calculate mean and standard deviation
    mean = np.mean(data)
    std = np.std(data)

    # Avoid division by zero
    if std == 0:
        return data - mean

    # Return normalized data (z-score)
    return (data - mean) / std

def clip_data(data, lower_quantile=0.01, upper_quantile=0.99):
    """
    Clip the data by removing extreme values based on specified quantiles.
    
    Args:
        data: EEG data with shape [channels, frames] or a single channel
        lower_quantile: Lower quantile threshold (default: 0.01, i.e., 1st percentile)
        upper_quantile: Upper quantile threshold (default: 0.99, i.e., 99th percentile)
    
    Returns:
        Clipped data with same shape as input
    """
    print(f"Clipping data between {lower_quantile:.2%} and {upper_quantile:.2%} quantiles...")
    
    # Check if input is a single channel or multi-channel
    is_single_channel = len(data.shape) == 1
    
    # Reshape single channel data for consistent processing
    if is_single_channel:
        data = data.reshape(1, -1)
    
    # Make a copy to avoid modifying the original data
    data_copy = data.copy()
    
    # Clip each channel individually based on its own quantiles
    for ch in range(data_copy.shape[0]):
        channel_data = data_copy[ch, :]
        lower_bound = np.quantile(channel_data, lower_quantile)
        upper_bound = np.quantile(channel_data, upper_quantile)
        
        # Print clipping info for this channel
        orig_range = (np.min(channel_data), np.max(channel_data))
        print(f"Channel {ch}: Original range [{orig_range[0]:.2f}, {orig_range[1]:.2f}], " 
              f"Clipping to [{lower_bound:.2f}, {upper_bound:.2f}]")
        
        # Apply clipping
        data_copy[ch, :] = np.clip(channel_data, lower_bound, upper_bound)
        
        # Calculate percentage of clipped values
        num_clipped_lower = np.sum(channel_data < lower_bound)
        num_clipped_upper = np.sum(channel_data > upper_bound)
        total_values = len(channel_data)
        pct_clipped = (num_clipped_lower + num_clipped_upper) / total_values * 100
        
        print(f"Channel {ch}: Clipped {pct_clipped:.2f}% of values "
              f"({num_clipped_lower} below, {num_clipped_upper} above)")
    
    # Return in the same shape as input
    if is_single_channel:
        return data_copy[0]
    
    return data_copy

def apply_bandpass_filter(data, sample_rate=500, low_freq=5.0, high_freq=45.0, order=4):
    """
    Apply a bandpass filter to keep only the specified EEG frequency range.

    Args:
        data: EEG data with shape [channels, frames] or a single channel
        sample_rate: Sampling rate in Hz (default: 500)
        low_freq: Lower cutoff frequency in Hz (default: 5.0)
        high_freq: Upper cutoff frequency in Hz (default: 45.0)
        order: Filter order (default: 4)

    Returns:
        Filtered data with same shape as input
    """
    print(f"Applying bandpass filter ({low_freq}-{high_freq} Hz)...")
    
    # Check if input is a single channel or multi-channel
    is_single_channel = len(data.shape) == 1

    # Reshape single channel data for consistent processing
    if is_single_channel:
        data = data.reshape(1, -1)

    # Make a copy to avoid modifying the original data
    data_copy = data.copy().astype(np.float64)
    
    # Ensure the data is in a reasonable range
    for ch in range(data_copy.shape[0]):
        # Remove DC offset for each channel
        data_copy[ch, :] = data_copy[ch, :] - np.mean(data_copy[ch, :])
    
    # Create filtered data array
    filtered_data = np.zeros_like(data_copy)
    
    try:
        # Normalize the frequencies to Nyquist frequency (half the sampling rate)
        nyq = 0.5 * sample_rate
        low = low_freq / nyq
        high = high_freq / nyq
        
        # Ensure frequencies are in valid range
        if low <= 0:
            low = 0.001  # Small positive value
            print(f"Warning: Low cutoff too small, adjusted to {low*nyq:.3f} Hz")
        if high >= 1:
            high = 0.999  # Just below Nyquist
            print(f"Warning: High cutoff too large, adjusted to {high*nyq:.3f} Hz")
        
        # Design Butterworth bandpass filter
        # Use safer sos (second-order sections) form instead of b, a coefficients
        sos = signal.butter(order, [low, high], btype='bandpass', output='sos')
        
        # Apply the filter to each channel
        for ch in range(data_copy.shape[0]):
            # Use sosfilt which is more numerically stable than filtfilt with high-order filters
            filtered_data[ch, :] = signal.sosfiltfilt(sos, data_copy[ch, :])
        
        print(f"Bandpass filter successfully applied.")
        
        # Verify filter effect by checking power in different bands
        ch = 0  # Just check the first channel
        f_orig, psd_orig = signal.welch(data_copy[ch, :], fs=sample_rate, nperseg=min(1024, data_copy.shape[1]//2))
        f_filt, psd_filt = signal.welch(filtered_data[ch, :], fs=sample_rate, nperseg=min(1024, data_copy.shape[1]//2))
        
        # Calculate and print power ratios in different bands
        idx_below = np.where(f_orig < low_freq)[0]
        idx_band = np.where((f_orig >= low_freq) & (f_orig <= high_freq))[0]
        idx_above = np.where(f_orig > high_freq)[0]
        
        if len(idx_below) > 0 and np.sum(psd_orig[idx_below]) > 0:
            below_ratio = np.sum(psd_filt[idx_below]) / np.sum(psd_orig[idx_below])
            print(f"Power below {low_freq}Hz reduced to {below_ratio:.1%} of original")
            
        if len(idx_band) > 0 and np.sum(psd_orig[idx_band]) > 0:
            band_ratio = np.sum(psd_filt[idx_band]) / np.sum(psd_orig[idx_band])
            print(f"Power in {low_freq}-{high_freq}Hz band is {band_ratio:.1%} of original")
            
        if len(idx_above) > 0 and np.sum(psd_orig[idx_above]) > 0:
            above_ratio = np.sum(psd_filt[idx_above]) / np.sum(psd_orig[idx_above])
            print(f"Power above {high_freq}Hz reduced to {above_ratio:.1%} of original")
    
    except Exception as e:
        print(f"Warning: Bandpass filter failed with error: {str(e)}")
        print("Returning original data...")
        filtered_data = data_copy  # Return the original data if filtering fails
    
    # Return in the same shape as input
    if is_single_channel:
        return filtered_data[0]

    return filtered_data

def plot_qq_channels(eeg_data, num_channels=6, output_path="plot.png", normalize=True):
    """
    Create a single Q-Q plot with multiple channels on the same graph.

    Args:
        eeg_data: EEG data with shape [channels, frames]
        num_channels: Number of channels to plot (default: 6)
        output_path: Path to save the plot
        normalize: Whether to normalize the data before plotting (default: True)
    """
    if eeg_data is None or eeg_data.shape[0] < 1:
        print("Not enough data to create Q-Q plots")
        return

    # Limit to the specified number of channels
    num_channels = min(num_channels, eeg_data.shape[0])

    # Create a single figure
    plt.figure(figsize=(12, 8))

    # Get a colormap for different channels
    colors = plt.cm.viridis(np.linspace(0, 1, num_channels))

    # Plot all channels on the same graph
    for i in range(num_channels):
        channel_data = eeg_data[i].copy()

        # Normalize data if requested
        if normalize:
            channel_data = normalize_channel(channel_data)

        # Get theoretical quantiles and ordered data for manual Q-Q plot
        osm = stats.probplot(channel_data, dist="norm", fit=False)
        theoretical_quantiles = osm[0]
        ordered_data = osm[1]

        # Plot with a different color and label for each channel
        plt.plot(theoretical_quantiles, ordered_data, 'o',
                 markersize=3, alpha=0.6, color=colors[i],
                 label=f'Channel {i}')

    # Add reference line
    if normalize:
        # For normalized data, reference line should be y=x
        plt.plot([-3, 3], [-3, 3], 'k--', label='Normal')
    else:
        # For unnormalized data, determine appropriate reference line
        min_x = min([min(stats.probplot(eeg_data[i], dist="norm", fit=False)[0])
                    for i in range(num_channels)])
        max_x = max([max(stats.probplot(eeg_data[i], dist="norm", fit=False)[0])
                    for i in range(num_channels)])
        plt.plot([min_x, max_x], [min_x, max_x], 'k--', label='Normal')

    # Add labels and title
    plt.xlabel('Theoretical Quantiles')
    plt.ylabel('Sample Quantiles')
    norm_text = "Normalized " if normalize else ""
    plt.title(f'Q-Q Plot for Multiple {norm_text}EEG Channels')
    plt.grid(True)
    plt.legend(loc='best')

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Q-Q plot saved to {output_path}")
    plt.close()

def calculate_channel_ranges(data):
    """
    Calculate the range (max - min) for each channel in the data.

    Args:
        data: EEG data with shape [channels, samples]

    Returns:
        Array of range values for each channel
    """
    if data is None or data.shape[1] < 2:
        return None

    # Calculate min and max for each channel
    channel_mins = np.min(data, axis=1)
    channel_maxs = np.max(data, axis=1)

    # Calculate range for each channel
    channel_ranges = channel_maxs - channel_mins

    return channel_ranges

def calculate_sliding_range(eeg_data, window_size=2500, step_size=None):
    """
    Calculate the range (max - min) for each channel in a sliding window.

    Args:
        eeg_data: EEG data with shape [channels, frames]
        window_size: Size of sliding window in samples (default: 2500 samples = 5s at 500Hz)
        step_size: Step size for the sliding window (default: window_size//5)

    Returns:
        Tuple containing (time_points, channel_ranges):
        - time_points: Array of time points (window centers) in samples
        - channel_ranges: Array with shape [channels, num_windows] containing max-min for each window
    """
    if eeg_data is None or eeg_data.shape[1] < window_size:
        return None, None

    # Set default step size if not provided
    if step_size is None:
        step_size = window_size // 5  # Default to 1/5 of window size

    num_channels = eeg_data.shape[0]
    num_frames = eeg_data.shape[1]

    # Calculate number of windows
    num_windows = (num_frames - window_size) // step_size + 1

    # Initialize arrays to store results
    time_points = np.zeros(num_windows)
    channel_ranges = np.zeros((num_channels, num_windows))

    # Calculate range for each window
    for w in range(num_windows):
        start_idx = w * step_size
        end_idx = start_idx + window_size
        center_idx = start_idx + window_size // 2

        # Store window center time point
        time_points[w] = center_idx

        # Get window data and calculate ranges for all channels
        window_data = eeg_data[:, start_idx:end_idx]
        window_ranges = calculate_channel_ranges(window_data)
        channel_ranges[:, w] = window_ranges

    return time_points, channel_ranges

def detect_flatlines(eeg_data, window_size=2500, threshold=0.005):
    """
    Detect flatlines in EEG channels where signal variation stays within a small threshold.

    Args:
        eeg_data: EEG data with shape [channels, frames]
        window_size: Size of windows to check for flatlines (default: 2500 samples = 5s at 500Hz)
        threshold: Maximum allowed variation in the window to be considered flat (default: 0.005)

    Returns:
        Numpy array with percentage of flatlined windows per channel
    """
    if eeg_data is None or eeg_data.shape[1] < window_size:
        return None

    num_channels = eeg_data.shape[0]
    num_windows = eeg_data.shape[1] // window_size

    # Initialize array to store flatline percentages
    flatline_percentages = np.zeros(num_channels)

    # Check each channel
    for ch in range(num_channels):
        channel_data = eeg_data[ch]
        flatlines_count = 0

        # Check each window
        for w in range(num_windows):
            start_idx = w * window_size
            end_idx = (w + 1) * window_size
            window_data = channel_data[start_idx:end_idx]

            # Calculate range of values in the window
            value_range = np.max(window_data) - np.min(window_data)  # Keep this simple for a single channel

            # Check if the range is below threshold (flatline)
            if value_range <= threshold:
                flatlines_count += 1

        # Calculate percentage
        if num_windows > 0:
            flatline_percentages[ch] = (flatlines_count / num_windows) * 100

    return flatline_percentages

def plot_sliding_range(eeg_data, num_channels=None, window_size=2500, step_size=None, sample_rate=500, output_path="plot.png"):
    """
    Create a plot showing the median signal range (max-min) across channels over time using a sliding window.

    Args:
        eeg_data: EEG data with shape [channels, frames]
        num_channels: Number of channels to analyze (default: None = all channels)
        window_size: Size of sliding window in samples (default: 2500 samples = 5s at 500Hz)
        step_size: Step size for the sliding window (default: window_size//5)
        sample_rate: Sampling rate in Hz (default: 500)
        output_path: Path to save the plot
    """
    if eeg_data is None:
        print("No data to analyze sliding range")
        return

    # Use all channels if num_channels not specified or limit to specified number
    if num_channels is None:
        num_channels = eeg_data.shape[0]
    else:
        num_channels = min(num_channels, eeg_data.shape[0])

    # Select channels to analyze
    data_to_analyze = eeg_data[:num_channels]

    # Calculate sliding range values
    time_points, channel_ranges = calculate_sliding_range(data_to_analyze, window_size, step_size)

    if time_points is None or channel_ranges is None:
        print("Unable to analyze sliding range - insufficient data")
        return

    # Convert time points to seconds
    time_seconds = time_points / sample_rate

    # Calculate median range across channels
    median_range = np.median(channel_ranges, axis=0)

    # Calculate quartiles across channels for each time point
    q1_range = np.percentile(channel_ranges, 25, axis=0)  # 25th percentile (1st quartile)
    q3_range = np.percentile(channel_ranges, 75, axis=0)  # 75th percentile (3rd quartile)

    # Create the plot
    plt.figure(figsize=(12, 6))

    # Plot median range as a line
    plt.plot(time_seconds, median_range, 'b-', linewidth=2)

    # Add interquartile range (Q1-Q3) as a shaded area
    plt.fill_between(time_seconds, q1_range, q3_range, color='b', alpha=0.2)

    # Add labels and title
    window_sec = window_size / sample_rate
    plt.xlabel('Time (seconds)')
    plt.ylabel('Signal Range (max - min)')
    plt.title(f'EEG Signal Range in {window_sec:.1f}s Sliding Window')
    plt.grid(True)

    # Add a legend
    plt.plot([], [], 'b-', linewidth=2, label='Median Range')
    plt.plot([], [], color='b', alpha=0.2, linewidth=10, label='Interquartile Range (25th-75th percentile)')
    plt.legend(loc='best')

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Sliding range plot saved to {output_path}")
    plt.close()

def plot_multi_segment_range(segments, num_channels, title_prefix="Segment", output_path="plot.png",
                          window_size=2500, sample_rate=500):
    """
    Create a single figure with sliding range plots for multiple segments.

    Args:
        segments: List of segmented EEG data
        num_channels: Number of channels to analyze
        title_prefix: Prefix for subplot titles
        output_path: Path to save the plot
        window_size: Size of sliding window in samples
        sample_rate: Sampling rate in Hz
    """
    num_segments = len(segments)
    if num_segments == 0:
        return

    # Calculate the grid dimensions
    grid_size = int(np.ceil(np.sqrt(num_segments)))
    rows = int(np.ceil(num_segments / grid_size))
    cols = grid_size

    # Create figure
    plt.figure(figsize=(cols*5, rows*4))

    # Create a subplot for each segment
    for i, segment in enumerate(segments):
        # Need a minimum amount of data for meaningful analysis
        if segment.shape[1] < window_size * 2:  # Need at least 2 windows
            ax = plt.subplot(rows, cols, i+1)
            ax.text(0.5, 0.5, 'Insufficient data for sliding range analysis',
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes)
            continue

        # Limit to the specified number of channels
        seg_num_channels = min(num_channels, segment.shape[0])
        segment_data = segment[:seg_num_channels]

        # Calculate sliding range
        step_size = max(window_size // 5, 1)  # Adjust step size for smaller segments
        time_points, channel_ranges = calculate_sliding_range(segment_data, window_size, step_size)

        if time_points is None or channel_ranges is None:
            ax = plt.subplot(rows, cols, i+1)
            ax.text(0.5, 0.5, 'Sliding range analysis failed',
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes)
            continue

        # Convert time points to seconds
        time_seconds = time_points / sample_rate

        # Calculate median range across channels
        median_range = np.median(channel_ranges, axis=0)

        # Create subplot
        ax = plt.subplot(rows, cols, i+1)

        # Plot median range as a line
        ax.plot(time_seconds, median_range, 'b-', linewidth=2)

        # Add interquartile range as a shaded area
        q1_range = np.percentile(channel_ranges, 25, axis=0)  # 25th percentile (1st quartile)
        q3_range = np.percentile(channel_ranges, 75, axis=0)  # 75th percentile (3rd quartile)
        ax.fill_between(time_seconds, q1_range, q3_range, color='b', alpha=0.2)

        # Add labels and title
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Signal Range')
        ax.set_title(f'{title_prefix} {i+1}')
        ax.grid(True)

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Multi-segment sliding range plots saved to {output_path}")
    plt.close()

def plot_flatlines(eeg_data, num_channels=None, window_size=2500, threshold=0.005, output_path="plot.png"):
    """
    Create a bar chart showing the percentage of time each channel is flat-lined.

    Args:
        eeg_data: EEG data with shape [channels, frames]
        num_channels: Number of channels to analyze (default: None = all channels)
        window_size: Size of windows to check for flatlines (default: 2500 samples = 5s at 500Hz)
        threshold: Maximum allowed variation to be considered flat (default: 0.005)
        output_path: Path to save the plot
    """
    if eeg_data is None:
        print("No data to analyze flatlines")
        return

    # Use all channels if num_channels not specified or limit to specified number
    if num_channels is None:
        num_channels = eeg_data.shape[0]
    else:
        num_channels = min(num_channels, eeg_data.shape[0])

    # Select channels to analyze
    data_to_analyze = eeg_data[:num_channels]

    # Calculate flatline percentages
    flatline_percentages = detect_flatlines(data_to_analyze, window_size, threshold)

    if flatline_percentages is None:
        print("Unable to analyze flatlines - insufficient data")
        return

    # Create the plot
    plt.figure(figsize=(12, 6))

    # Create bar chart
    channel_indices = np.arange(num_channels)
    plt.bar(channel_indices, flatline_percentages, color='firebrick')

    # Add labels and title
    plt.xlabel('Channel Index')
    plt.ylabel('Flatline Percentage (%)')
    plt.title(f'EEG Channel Flatline Percentage (±{threshold} threshold)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(channel_indices)

    # Add values on top of bars
    for i, percentage in enumerate(flatline_percentages):
        plt.text(i, percentage + 1, f'{percentage:.1f}%', ha='center')

    # Add a horizontal line at 0%
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Flatline analysis plot saved to {output_path}")
    plt.close()

def plot_multi_segment_flatlines(segments, num_channels, title_prefix="Segment", output_path="plot.png",
                              window_size=2500, threshold=0.005):
    """
    Create a single figure with flatline percentages for multiple segments.

    Args:
        segments: List of segmented EEG data
        num_channels: Number of channels to analyze
        title_prefix: Prefix for subplot titles
        output_path: Path to save the plot
        window_size: Size of windows to check for flatlines
        threshold: Maximum allowed variation to be considered flat
    """
    num_segments = len(segments)
    if num_segments == 0:
        return

    # Calculate the grid dimensions
    grid_size = int(np.ceil(np.sqrt(num_segments)))
    rows = int(np.ceil(num_segments / grid_size))
    cols = grid_size

    # Create figure
    plt.figure(figsize=(cols*5, rows*4))

    # Create a subplot for each segment
    for i, segment in enumerate(segments):
        # Need a minimum amount of data for meaningful analysis
        if segment.shape[1] < window_size:
            continue

        # Limit to the specified number of channels
        seg_num_channels = min(num_channels, segment.shape[0])
        segment_data = segment[:seg_num_channels]

        # Calculate flatline percentages
        flatline_percentages = detect_flatlines(segment_data, window_size, threshold)

        if flatline_percentages is None:
            continue

        # Create subplot
        ax = plt.subplot(rows, cols, i+1)

        # Create bar chart
        channel_indices = np.arange(seg_num_channels)
        ax.bar(channel_indices, flatline_percentages, color='firebrick')

        # Add labels and title
        ax.set_xlabel('Channel Index')
        ax.set_ylabel('Flatline %')
        ax.set_title(f'{title_prefix} {i+1}')
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        # Only show a subset of ticks for readability
        tick_step = max(1, seg_num_channels // 10)
        ax.set_xticks(channel_indices[::tick_step])

        # Add a horizontal line at 0%
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Multi-segment flatline analysis plots saved to {output_path}")
    plt.close()

def analyze_fooof(freqs, psd, freq_range=(3, 45), peak_width_limits=(2, 8), 
               max_n_peaks=6, min_peak_height=0.05, peak_threshold=2.0, verbose=False):
    """
    Analyze power spectrum using FOOOF to separate periodic and aperiodic components.
    
    Args:
        freqs: Frequency vector
        psd: Power spectral density vector
        freq_range: Frequency range to fit (default: (3, 45))
        peak_width_limits: Limits for peak width (default: (2, 8))
        max_n_peaks: Maximum number of peaks to fit (default: 6)
        min_peak_height: Minimum peak height to detect (default: 0.05)
        peak_threshold: Threshold for peak detection (default: 2.0)
        verbose: Whether to print FOOOF model information (default: False)
        
    Returns:
        Dictionary containing FOOOF model and component spectra
    """
    if not FOOOF_AVAILABLE:
        return None
    
    # Initialize FOOOF model
    fm = FOOOF(peak_width_limits=peak_width_limits, max_n_peaks=max_n_peaks, 
               min_peak_height=min_peak_height, peak_threshold=peak_threshold, verbose=verbose)
    
    # Fit the model
    fm.fit(freqs, psd, freq_range)
    
    # Get model components
    ap_fit = fm.get_model(component='aperiodic')  # 1/f-like aperiodic component
    peak_fit = fm.get_model(component='peak')     # Oscillatory/peak component
    model_fit = fm.get_model(component='full')    # Full model fit (aperiodic + peak)
    
    # Get individual peak parameters (CF: center frequency, PW: peak width, BW: bandwidth)
    peaks = fm.get_params('peak_params')
    
    # Define standard EEG frequency bands
    bands = Bands({'theta': [4, 8],
                   'alpha': [8, 13], 
                   'beta': [13, 30],
                   'gamma': [30, 45]})
    
    # Extract peaks in each band
    band_peaks = {}
    for band_name, band_def in bands:
        # get_band_peak_fg expects a FOOOFGroup object, not just the parameters
        # We'll handle this more safely by just using the peaks we already have
        band_peaks[band_name] = []
        if peaks.size > 0:
            # For each peak, check if it's in this band and add it if so
            for peak in peaks:
                cf, pw, bw = peak
                if band_def[0] <= cf <= band_def[1]:
                    band_peaks[band_name].append(peak)
    
    # Return all results
    return {
        'model': fm,
        'aperiodic': ap_fit,
        'periodic': peak_fit,  # Renamed but keeping key as 'periodic' for backward compatibility
        'full_model': model_fit,
        'peaks': peaks,
        'band_peaks': band_peaks,
        'r_squared': fm.r_squared_,
        'error': fm.error_,
        'freqs': freqs,
        'original_psd': psd
    }

def plot_fooof_spectrum(eeg_data, num_channels=6, sample_rate=500, output_path="plot.png", 
                     log_scale=True, log_freq=False, apply_filter=True, low_freq=3.0, high_freq=45.0):
    """
    Create FOOOF power spectrum plots for the first N channels, showing the aperiodic and periodic components.
    
    Args:
        eeg_data: EEG data with shape [channels, frames]
        num_channels: Number of channels to plot (default: 6)
        sample_rate: Sampling rate in Hz (default: 500)
        output_path: Path to save the plot
        log_scale: Whether to use logarithmic scale for power (y-axis) (default: True)
        log_freq: Whether to use logarithmic scale for frequency (x-axis) (default: False)
        apply_filter: Whether to apply bandpass filter before analysis (default: True)
        low_freq: Lower cutoff frequency in Hz for analysis (default: 3.0)
        high_freq: Upper cutoff frequency in Hz for analysis (default: 45.0)
    """
    if not FOOOF_AVAILABLE:
        print("FOOOF package not available. Please install with: pip install fooof")
        return
    
    if eeg_data is None or eeg_data.shape[0] < 1:
        print("Not enough data to create FOOOF spectrum plots")
        return
    
    # Limit to the specified number of channels
    num_channels = min(num_channels, eeg_data.shape[0])
    
    # Create a figure with subplots for each channel
    fig, axes = plt.subplots(num_channels, 1, figsize=(12, 3*num_channels), sharex=True)
    if num_channels == 1:
        axes = [axes]  # Make it a list for consistent indexing
    
    # Get a colormap for different components
    colors = {
        'original': 'black',
        'model': 'red',
        'aperiodic': 'blue',
        'periodic': 'green',
        'bands': plt.cm.tab10
    }
    
    # EEG frequency bands (for visualization)
    bands = [
        ('Theta', (4, 8), colors['bands'](0)),
        ('Alpha', (8, 13), colors['bands'](1)),
        ('Beta', (13, 30), colors['bands'](2)),
        ('Gamma', (30, 45), colors['bands'](3))
    ]
    
    # Apply bandpass filter if requested
    filtered_data = eeg_data.copy()
    if apply_filter:
        filtered_data = apply_bandpass_filter(filtered_data, sample_rate, low_freq, high_freq)
        filter_text = f" (Filtered: {low_freq}-{high_freq} Hz)"
    else:
        filter_text = ""
    
    # Analyze each channel
    for i in range(num_channels):
        ax = axes[i]
        channel_data = filtered_data[i]
        
        # Use Welch's method to calculate power spectral density
        freqs, psd = signal.welch(channel_data, sample_rate, nperseg=min(4096, len(channel_data)))
        
        # Restrict to frequency range of interest
        mask = (freqs >= low_freq) & (freqs <= high_freq)
        freqs_masked = freqs[mask]
        psd_masked = psd[mask]
        
        # Analyze PSD with FOOOF
        fooof_results = analyze_fooof(freqs_masked, psd_masked, freq_range=(low_freq, high_freq))
        
        # Plot original PSD with appropriate scale
        if log_scale and log_freq:
            # Log-log plot
            ax.loglog(freqs_masked, psd_masked, label='Original PSD', color=colors['original'], linewidth=2, alpha=0.7)
            ax.loglog(freqs_masked, fooof_results['full_model'], label='FOOOF Model', color=colors['model'], linewidth=2, alpha=0.7)
            ax.loglog(freqs_masked, fooof_results['aperiodic'], label='Aperiodic (1/f)', color=colors['aperiodic'], linewidth=2, alpha=0.7, linestyle='--')
            ax.loglog(freqs_masked, fooof_results['periodic'], label='Periodic', color=colors['periodic'], linewidth=2, alpha=0.7)
        elif log_scale:
            # Semi-log plot (y-axis only)
            ax.semilogy(freqs_masked, psd_masked, label='Original PSD', color=colors['original'], linewidth=2, alpha=0.7)
            ax.semilogy(freqs_masked, fooof_results['full_model'], label='FOOOF Model', color=colors['model'], linewidth=2, alpha=0.7)
            ax.semilogy(freqs_masked, fooof_results['aperiodic'], label='Aperiodic (1/f)', color=colors['aperiodic'], linewidth=2, alpha=0.7, linestyle='--')
            ax.semilogy(freqs_masked, fooof_results['periodic'], label='Periodic', color=colors['periodic'], linewidth=2, alpha=0.7)
        elif log_freq:
            # Semi-log plot (x-axis only)
            ax.semilogx(freqs_masked, psd_masked, label='Original PSD', color=colors['original'], linewidth=2, alpha=0.7)
            ax.semilogx(freqs_masked, fooof_results['full_model'], label='FOOOF Model', color=colors['model'], linewidth=2, alpha=0.7)
            ax.semilogx(freqs_masked, fooof_results['aperiodic'], label='Aperiodic (1/f)', color=colors['aperiodic'], linewidth=2, alpha=0.7, linestyle='--')
            ax.semilogx(freqs_masked, fooof_results['periodic'], label='Periodic', color=colors['periodic'], linewidth=2, alpha=0.7)
        else:
            # Linear plot
            ax.plot(freqs_masked, psd_masked, label='Original PSD', color=colors['original'], linewidth=2, alpha=0.7)
            ax.plot(freqs_masked, fooof_results['full_model'], label='FOOOF Model', color=colors['model'], linewidth=2, alpha=0.7)
            ax.plot(freqs_masked, fooof_results['aperiodic'], label='Aperiodic (1/f)', color=colors['aperiodic'], linewidth=2, alpha=0.7, linestyle='--')
            ax.plot(freqs_masked, fooof_results['periodic'], label='Periodic', color=colors['periodic'], linewidth=2, alpha=0.7)
        
        # Add colored backgrounds for frequency bands
        y_min, y_max = ax.get_ylim()
        for band_name, (band_low, band_high), band_color in bands:
            if band_high >= low_freq and band_low <= high_freq:  # Only show bands in our display range
                ax.axvspan(max(band_low, low_freq), min(band_high, high_freq), alpha=0.1, color=band_color)
                # Add text labels at the top of the first plot only
                if i == 0:
                    ax.text((max(band_low, low_freq) + min(band_high, high_freq))/2, y_max*0.95, band_name,
                           horizontalalignment='center', verticalalignment='top',
                           fontsize=8, color='black')
        
        # Mark fitted peaks
        for peak in fooof_results['peaks']:
            if peak[0] >= low_freq and peak[0] <= high_freq:  # Only show peaks in our display range
                peak_freq, peak_power, peak_width = peak
                
                # Different plotting methods based on axis scales
                if log_scale and log_freq:
                    # For log-log plot 
                    ax.plot(peak_freq, 10**peak_power, 'o', markersize=8, alpha=0.7, color=colors['periodic'])
                elif log_scale:
                    # For semi-log plot (y-axis only)
                    ax.plot(peak_freq, 10**peak_power, 'o', markersize=8, alpha=0.7, color=colors['periodic'])
                elif log_freq:
                    # For semi-log plot (x-axis only)
                    ax.plot(peak_freq, peak_power, 'o', markersize=8, alpha=0.7, color=colors['periodic'])
                else:
                    # For linear plot
                    ax.plot(peak_freq, peak_power, 'o', markersize=8, alpha=0.7, color=colors['periodic'])
        
        # Add labels and grid
        ax.set_xlabel('Frequency (Hz)' if i == num_channels-1 else '')
        
        # Set y-axis label based on scaling
        if log_scale:
            ax.set_ylabel('Log Power')
        else:
            ax.set_ylabel('Power')
            
        # Update x-axis label if using log frequency
        if log_freq and i == num_channels-1:
            ax.set_xlabel('Log Frequency (Hz)')
            
        ax.set_title(f'Channel {i} - FOOOF Analysis (R²={fooof_results["r_squared"]:.3f})')
        
        # Use appropriate grid type based on axis scales
        if log_scale or log_freq:
            ax.grid(True, which="both")  # Show major and minor grid for any log scale
        else:
            ax.grid(True, which="major")  # Only major grid for linear scale
        
        # Add legend for first channel only
        if i == 0:
            ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"FOOOF spectrum analysis saved to {output_path}")
    plt.close()

def plot_power_spectrum(eeg_data, num_channels=6, sample_rate=500, output_path="plot.png",
                    log_scale=True, apply_filter=True, low_freq=2.0, high_freq=45.0):
    """
    Create power spectrum plots for the first N channels.

    Args:
        eeg_data: EEG data with shape [channels, frames]
        num_channels: Number of channels to plot (default: 6)
        sample_rate: Sampling rate in Hz (default: 500)
        output_path: Path to save the plot
        log_scale: Whether to use logarithmic scale for power (y-axis) (default: True)
        apply_filter: Whether to apply bandpass filter before analysis (default: True)
        low_freq: Lower cutoff frequency in Hz for bandpass filter (default: 2.0)
        high_freq: Upper cutoff frequency in Hz for bandpass filter (default: 45.0)
    """
    if eeg_data is None or eeg_data.shape[0] < 1:
        print("Not enough data to create power spectrum plots")
        return

    # Limit to the specified number of channels
    num_channels = min(num_channels, eeg_data.shape[0])

    # Create a figure
    plt.figure(figsize=(12, 8))

    # Get a colormap for different channels
    colors = plt.cm.viridis(np.linspace(0, 1, num_channels))

    # Apply bandpass filter if requested
    filtered_data = eeg_data.copy()
    if apply_filter:
        filtered_data = apply_bandpass_filter(filtered_data, sample_rate, low_freq, high_freq)
        filter_text = f" (Filtered: {low_freq}-{high_freq} Hz)"
    else:
        filter_text = ""

    # Calculate and plot power spectrum for each channel
    for i in range(num_channels):
        channel_data = filtered_data[i]

        # Use Welch's method to calculate power spectral density
        freqs, psd = signal.welch(channel_data, sample_rate, nperseg=min(4096, len(channel_data)))

        # Plot only from 5 Hz to 70 Hz (or up to high_freq if it's lower)
        upper_freq = min(70, high_freq + 10)  # Show a bit above the filter cutoff
        mask = (freqs >= max(1, low_freq - 1)) & (freqs <= upper_freq)

        if log_scale:
            # Use semilogy for log scale on y-axis
            plt.semilogy(freqs[mask], psd[mask], linewidth=2, alpha=0.7, color=colors[i], label=f'Channel {i}')
        else:
            # Linear scale
            plt.plot(freqs[mask], psd[mask], linewidth=2, alpha=0.7, color=colors[i], label=f'Channel {i}')

    # Add labels and title
    plt.xlabel('Frequency (Hz)')
    scale_text = "Log " if log_scale else ""
    plt.ylabel(f'{scale_text}Power Spectral Density')
    plt.title(f'EEG Power Spectrum{filter_text}')
    plt.grid(True, which="both" if log_scale else "major")

    # Add minor grid lines for log scale
    if log_scale:
        plt.grid(True, which="minor", axis="y", linestyle=":", alpha=0.4)

    plt.legend(loc='best')

    # Show common EEG frequency bands
    band_colors = ['#e6194B', '#f58231', '#ffe119', '#3cb44b', '#4363d8']
    band_names = ['Theta', 'Alpha', 'Beta', 'Low Gamma', 'High Gamma']
    band_ranges = [(4, 8), (8, 13), (13, 30), (30, 50), (50, 70)]

    # Add colored backgrounds for frequency bands
    y_min, y_max = plt.ylim()
    for (low, high), color, name in zip(band_ranges, band_colors, band_names):
        if high >= 5:  # Only show bands that overlap with our display range
            plt.axvspan(max(5, low), min(high, 70), alpha=0.1, color=color)
            # Add text label at the top of each band
            plt.text((max(5, low) + min(high, 70))/2, y_max*0.95, name,
                     horizontalalignment='center', verticalalignment='top',
                     fontsize=8, color='black')

    plt.tight_layout()
    plt.savefig(output_path)
    log_text = "logarithmic " if log_scale else ""
    print(f"Power spectrum plot (with {log_text}y-axis) saved to {output_path}")
    plt.close()

def calculate_mutual_information_matrix(data):
    """
    Calculate the mutual information matrix for a set of variables.
    
    Args:
        data: Data array with shape [variables, samples]
        
    Returns:
        Mutual information matrix with shape [variables, variables]
    """
    n_variables = data.shape[0]
    mi_matrix = np.zeros((n_variables, n_variables))
    
    # Normalize the data to improve numerical stability
    normalized_data = np.copy(data)
    for i in range(n_variables):
        mean = np.mean(data[i])
        std = np.std(data[i])
        if std > 0:
            normalized_data[i] = (data[i] - mean) / std
    
    # Calculate mutual information between each pair of variables
    for i in range(n_variables):
        x = normalized_data[i].reshape(-1, 1)
        
        # Diagonal elements (self-information)
        mi_matrix[i, i] = 1.0
        
        # Off-diagonal elements
        for j in range(i+1, n_variables):
            y = normalized_data[j]
            # Calculate mutual information using scikit-learn
            mi = mutual_info_regression(x, y)[0]
            
            # Normalize MI to a 0-1 scale (approximately)
            # A value of 0 means no mutual information, 1 means high mutual information
            normalized_mi = min(1.0, mi / 2.0)  # Heuristic normalization
            
            # Mutual information is symmetric
            mi_matrix[i, j] = normalized_mi
            mi_matrix[j, i] = normalized_mi
    
    return mi_matrix

def plot_mutual_information_matrix(eeg_data, num_channels=None, output_path="plot.png"):
    """
    Create a mutual information matrix plot showing relationships between channels.

    Args:
        eeg_data: EEG data with shape [channels, frames]
        num_channels: Number of channels to include (default: None = all channels)
        output_path: Path to save the plot
    """
    if eeg_data is None:
        print("No data to create mutual information matrix")
        return

    # Use all channels if num_channels not specified or limit to specified number
    if num_channels is None:
        num_channels = eeg_data.shape[0]
    else:
        num_channels = min(num_channels, eeg_data.shape[0])

    # Select the channels to analyze
    data_to_analyze = eeg_data[:num_channels]

    # Calculate mutual information matrix
    print("Calculating mutual information matrix (this may take a moment)...")
    mi_matrix = calculate_mutual_information_matrix(data_to_analyze)

    # Create the plot
    plt.figure(figsize=(10, 8))

    # Use a colormap for mutual information (0 to 1)
    cmap = plt.cm.viridis

    # Create heatmap with a colorbar
    im = plt.imshow(mi_matrix, cmap=cmap, vmin=0, vmax=1)
    plt.colorbar(im, label='Mutual Information (normalized)')

    # Add labels and title
    plt.title('EEG Channel Mutual Information Matrix')
    plt.xlabel('Channel Index')
    plt.ylabel('Channel Index')

    # Add channel indices as ticks
    channel_indices = range(num_channels)
    plt.xticks(channel_indices)
    plt.yticks(channel_indices)

    # Add grid lines to separate cells
    plt.grid(False)

    # Add mutual information values in each cell for readability
    for i in range(num_channels):
        for j in range(num_channels):
            text_color = 'white' if mi_matrix[i, j] > 0.7 else 'black'
            plt.text(j, i, f'{mi_matrix[i, j]:.2f}',
                     ha='center', va='center', color=text_color)

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Mutual information matrix saved to {output_path}")
    plt.close()

def plot_correlation_matrix(eeg_data, num_channels=None, output_path="plot.png"):
    """
    Create a correlation matrix plot showing relationships between channels.

    Args:
        eeg_data: EEG data with shape [channels, frames]
        num_channels: Number of channels to include (default: None = all channels)
        output_path: Path to save the plot
    """
    if eeg_data is None:
        print("No data to create correlation matrix")
        return

    # Use all channels if num_channels not specified or limit to specified number
    if num_channels is None:
        num_channels = eeg_data.shape[0]
    else:
        num_channels = min(num_channels, eeg_data.shape[0])

    # Select the channels to analyze
    data_to_analyze = eeg_data[:num_channels]

    # Calculate correlation matrix
    corr_matrix = np.corrcoef(data_to_analyze)

    # Create the plot
    plt.figure(figsize=(10, 8))

    # Use a colormap that highlights positive and negative correlations
    cmap = plt.cm.RdBu_r

    # Create heatmap with a colorbar
    im = plt.imshow(corr_matrix, cmap=cmap, vmin=-1, vmax=1)
    plt.colorbar(im, label='Correlation Coefficient')

    # Add labels and title
    plt.title('EEG Channel Correlation Matrix')
    plt.xlabel('Channel Index')
    plt.ylabel('Channel Index')

    # Add channel indices as ticks
    channel_indices = range(num_channels)
    plt.xticks(channel_indices)
    plt.yticks(channel_indices)

    # Add grid lines to separate cells
    plt.grid(False)

    # Add correlation values in each cell for readability
    for i in range(num_channels):
        for j in range(num_channels):
            text_color = 'white' if abs(corr_matrix[i, j]) > 0.7 else 'black'
            plt.text(j, i, f'{corr_matrix[i, j]:.2f}',
                     ha='center', va='center', color=text_color)

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Correlation matrix saved to {output_path}")
    plt.close()

def segment_data(eeg_data, num_segments):
    """
    Divide EEG data into equal segments.

    Args:
        eeg_data: EEG data with shape [channels, frames]
        num_segments: Number of segments to divide the data into

    Returns:
        List of segmented EEG data arrays
    """
    if eeg_data is None or num_segments < 1:
        return []

    # Calculate segment length
    total_frames = eeg_data.shape[1]
    segment_length = total_frames // num_segments

    # Create segments
    segments = []
    for i in range(num_segments):
        start_idx = i * segment_length
        end_idx = (i + 1) * segment_length if i < num_segments - 1 else total_frames
        segment = eeg_data[:, start_idx:end_idx]
        segments.append(segment)

    return segments

def run_analysis_function(eeg_data, analysis_func, **kwargs):
    """
    Run a specified analysis function on EEG data.
    This function allows for easily swapping different analysis methods.

    Args:
        eeg_data: EEG data with shape [channels, frames]
        analysis_func: Function to run on the data
        **kwargs: Additional arguments to pass to the analysis function
    """
    if eeg_data is None:
        print("No data to analyze")
        return

    # Run the specified analysis function
    return analysis_func(eeg_data, **kwargs)

def plot_multi_segment_std(segments, num_channels, title_prefix="Segment", output_path="plot.png"):
    """
    Create a single figure with standard deviation plots for multiple segments.

    Args:
        segments: List of segmented EEG data
        num_channels: Number of channels to analyze
        title_prefix: Prefix for subplot titles
        output_path: Path to save the plot
    """
    num_segments = len(segments)
    if num_segments == 0:
        return

    # Calculate the grid dimensions
    grid_size = int(np.ceil(np.sqrt(num_segments)))
    rows = int(np.ceil(num_segments / grid_size))
    cols = grid_size

    # Create figure
    plt.figure(figsize=(cols*5, rows*4))

    # Create a subplot for each segment
    for i, segment in enumerate(segments):
        # Calculate standard deviation for this segment
        segment_std = analyze_channel_std(segment, num_channels)

        # Create subplot
        plt.subplot(rows, cols, i+1)

        # Plot standard deviation bars
        channel_indices = np.arange(len(segment_std))
        plt.bar(channel_indices, segment_std, color='steelblue')
        plt.xlabel('Channel Index')
        plt.ylabel('Standard Deviation')
        plt.title(f'{title_prefix} {i+1}')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.xticks(channel_indices[::2])  # Show every other channel for clarity

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Multi-segment standard deviation plot saved to {output_path}")
    plt.close()

def plot_multi_segment_qq(segments, num_channels, title_prefix="Segment", output_path="plot.png"):
    """
    Create a single figure with Q-Q plots for multiple segments.

    Args:
        segments: List of segmented EEG data
        num_channels: Number of channels to analyze
        title_prefix: Prefix for subplot titles
        output_path: Path to save the plot
    """
    num_segments = len(segments)
    if num_segments == 0:
        return

    # Calculate the grid dimensions
    grid_size = int(np.ceil(np.sqrt(num_segments)))
    rows = int(np.ceil(num_segments / grid_size))
    cols = grid_size

    # Create figure
    plt.figure(figsize=(cols*5, rows*4))

    # Get a colormap for different channels
    colors = plt.cm.viridis(np.linspace(0, 1, num_channels))

    # Create a subplot for each segment
    for i, segment in enumerate(segments):
        # Create subplot
        ax = plt.subplot(rows, cols, i+1)

        # Plot all channels on the same graph
        for j in range(min(num_channels, segment.shape[0])):
            channel_data = segment[j]

            # Get theoretical quantiles and ordered data for manual Q-Q plot
            osm = stats.probplot(channel_data, dist="norm", fit=False)
            theoretical_quantiles = osm[0]
            ordered_data = osm[1]

            # Plot with a different color for each channel
            ax.plot(theoretical_quantiles, ordered_data, 'o',
                   markersize=2, alpha=0.5, color=colors[j])

        # Add reference line
        min_x = min([min(stats.probplot(segment[j], dist="norm", fit=False)[0])
                     for j in range(min(num_channels, segment.shape[0]))])
        max_x = max([max(stats.probplot(segment[j], dist="norm", fit=False)[0])
                     for j in range(min(num_channels, segment.shape[0]))])
        ax.plot([min_x, max_x], [min_x, max_x], 'k--')

        # Add labels and title
        ax.set_xlabel('Theoretical Quantiles')
        ax.set_ylabel('Sample Quantiles')
        ax.set_title(f'{title_prefix} {i+1}')
        ax.grid(True)

    # Add a single legend at the bottom of the figure
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[j], markersize=8, alpha=0.7, label=f'Ch {j}')
              for j in range(min(num_channels, 6))]  # Show first 6 channels in legend
    plt.figlegend(handles=handles, loc='lower center', ncol=min(num_channels, 6), bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Make room for legend
    plt.savefig(output_path)
    print(f"Multi-segment Q-Q plot saved to {output_path}")
    plt.close()

def plot_multi_segment_fooof(segments, num_channels, title_prefix="Segment", output_path="plot.png", 
                         sample_rate=500, log_scale=True, apply_filter=True, low_freq=3.0, high_freq=45.0):
    """
    Create a single figure with FOOOF power spectral density plots for multiple segments.
    
    Args:
        segments: List of segmented EEG data
        num_channels: Number of channels to analyze
        title_prefix: Prefix for subplot titles
        output_path: Path to save the plot
        sample_rate: Sampling rate in Hz
        log_scale: Whether to use logarithmic scale for power (y-axis)
        apply_filter: Whether to apply bandpass filter before analysis
        low_freq: Lower cutoff frequency in Hz for analysis
        high_freq: Upper cutoff frequency in Hz for analysis
    """
    if not FOOOF_AVAILABLE:
        print("FOOOF package not available. Please install with: pip install fooof")
        return
    
    num_segments = len(segments)
    if num_segments == 0:
        return
    
    # For multi-segment FOOOF analysis, we'll focus on just one channel per segment
    # to avoid an excessively large figure
    channel_to_analyze = 0  # Use the first channel for all segments
    
    # Calculate the grid dimensions
    grid_size = int(np.ceil(np.sqrt(num_segments)))
    rows = int(np.ceil(num_segments / grid_size))
    cols = grid_size
    
    # Create figure
    plt.figure(figsize=(cols*5, rows*4))
    
    # Colors for different components
    colors = {
        'original': 'black',
        'model': 'red',
        'aperiodic': 'blue',
        'periodic': 'green',
        'bands': plt.cm.tab10
    }
    
    # EEG frequency bands (for visualization)
    bands = [
        ('Theta', (4, 8), colors['bands'](0)),
        ('Alpha', (8, 13), colors['bands'](1)),
        ('Beta', (13, 30), colors['bands'](2)),
        ('Gamma', (30, 45), colors['bands'](3))
    ]
    
    # Create a subplot for each segment
    for i, segment in enumerate(segments):
        # Apply bandpass filter if requested
        filtered_segment = segment.copy()
        if apply_filter:
            filtered_segment = apply_bandpass_filter(filtered_segment, sample_rate, low_freq, high_freq)
            filter_text = f" (Filtered: {low_freq}-{high_freq} Hz)"
        else:
            filter_text = ""
        
        # Create subplot
        ax = plt.subplot(rows, cols, i+1)
        
        # Use data from the first channel
        channel_data = filtered_segment[channel_to_analyze]
        
        # Calculate PSD
        freqs, psd = signal.welch(channel_data, sample_rate, nperseg=min(2048, segment.shape[1]))
        
        # Restrict to frequency range of interest
        mask = (freqs >= low_freq) & (freqs <= high_freq)
        freqs_masked = freqs[mask]
        psd_masked = psd[mask]
        
        # Analyze with FOOOF
        fooof_results = analyze_fooof(freqs_masked, psd_masked, freq_range=(low_freq, high_freq))
        if fooof_results is None:
            ax.text(0.5, 0.5, 'FOOOF analysis failed or unavailable',
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes)
            continue
            
        # Plot curves
        if log_scale:
            ax.semilogy(freqs_masked, psd_masked, label='Original', color=colors['original'], linewidth=1.5, alpha=0.7)
            ax.semilogy(freqs_masked, fooof_results['full_model'], label='Model', color=colors['model'], linewidth=1.5, alpha=0.7)
            ax.semilogy(freqs_masked, fooof_results['aperiodic'], label='Aperiodic', color=colors['aperiodic'], linewidth=1.5, alpha=0.7, linestyle='--')
            ax.semilogy(freqs_masked, fooof_results['periodic'], label='Periodic', color=colors['periodic'], linewidth=1.5, alpha=0.7)
        else:
            ax.plot(freqs_masked, psd_masked, label='Original', color=colors['original'], linewidth=1.5, alpha=0.7)
            ax.plot(freqs_masked, fooof_results['full_model'], label='Model', color=colors['model'], linewidth=1.5, alpha=0.7)
            ax.plot(freqs_masked, fooof_results['aperiodic'], label='Aperiodic', color=colors['aperiodic'], linewidth=1.5, alpha=0.7, linestyle='--')
            ax.plot(freqs_masked, fooof_results['periodic'], label='Periodic', color=colors['periodic'], linewidth=1.5, alpha=0.7)
        
        # Add frequency band backgrounds
        y_min, y_max = ax.get_ylim()
        for band_name, (band_low, band_high), band_color in bands:
            if band_high >= low_freq and band_low <= high_freq:  # Only show bands in our display range
                ax.axvspan(max(band_low, low_freq), min(band_high, high_freq), alpha=0.1, color=band_color)
        
        # Mark fitted peaks
        for peak in fooof_results['peaks']:
            if peak[0] >= low_freq and peak[0] <= high_freq:  # Only show peaks in our display range
                peak_freq, peak_power, peak_width = peak
                if log_scale:
                    ax.plot(peak_freq, 10**peak_power, 'o', markersize=6, alpha=0.7, color=colors['periodic'])
                else:
                    ax.plot(peak_freq, peak_power, 'o', markersize=6, alpha=0.7, color=colors['periodic'])
        
        # Add title and labels
        ax.set_title(f'{title_prefix} {i+1} (R²={fooof_results["r_squared"]:.3f})')
        
        # Only add labels to subplots on the left and bottom edges
        if i % cols == 0:  # Left edge
            ax.set_ylabel('Power' if not log_scale else 'Log Power')
        if i >= (rows-1) * cols:  # Bottom edge
            ax.set_xlabel('Frequency (Hz)')
        
        # Add grid
        ax.grid(True, which="both" if log_scale else "major")
        
        # Add legend to first plot only
        if i == 0:
            ax.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Multi-segment FOOOF analysis saved to {output_path}")
    plt.close()

def plot_multi_segment_psd(segments, num_channels, title_prefix="Segment", output_path="plot.png",
                        sample_rate=500, log_scale=True, apply_filter=True, low_freq=2.0, high_freq=45.0):
    """
    Create a single figure with power spectral density plots for multiple segments.

    Args:
        segments: List of segmented EEG data
        num_channels: Number of channels to analyze
        title_prefix: Prefix for subplot titles
        output_path: Path to save the plot
        sample_rate: Sampling rate in Hz
        log_scale: Whether to use logarithmic scale for power (y-axis)
        apply_filter: Whether to apply bandpass filter before analysis
        low_freq: Lower cutoff frequency in Hz for bandpass filter
        high_freq: Upper cutoff frequency in Hz for bandpass filter
    """
    num_segments = len(segments)
    if num_segments == 0:
        return

    # Calculate the grid dimensions
    grid_size = int(np.ceil(np.sqrt(num_segments)))
    rows = int(np.ceil(num_segments / grid_size))
    cols = grid_size

    # Create figure
    plt.figure(figsize=(cols*5, rows*4))

    # Get a colormap for different channels
    colors = plt.cm.viridis(np.linspace(0, 1, num_channels))

    # Common EEG frequency bands
    band_colors = ['#e6194B', '#f58231', '#ffe119', '#3cb44b', '#4363d8']
    band_names = ['Theta', 'Alpha', 'Beta', 'Low Gamma', 'High Gamma']
    band_ranges = [(4, 8), (8, 13), (13, 30), (30, 50), (50, 70)]

    # Create a subplot for each segment
    for i, segment in enumerate(segments):
        # Create subplot
        ax = plt.subplot(rows, cols, i+1)

        # Apply bandpass filter if requested
        filtered_segment = segment.copy()
        if apply_filter:
            filtered_segment = apply_bandpass_filter(filtered_segment, sample_rate, low_freq, high_freq)
            filter_text = f" (Filtered: {low_freq}-{high_freq} Hz)"
        else:
            filter_text = ""

        # Calculate and plot power spectrum for each channel
        for j in range(min(num_channels, segment.shape[0])):
            channel_data = filtered_segment[j]

            # Use Welch's method to calculate power spectral density
            freqs, psd = signal.welch(channel_data, sample_rate, nperseg=min(2048, segment.shape[1]))

            # Plot frequency range based on filter settings
            upper_freq = min(70, high_freq + 10)  # Show a bit above the filter cutoff
            mask = (freqs >= max(1, low_freq - 1)) & (freqs <= upper_freq)

            if log_scale:
                # Use semilogy for log scale on y-axis
                ax.semilogy(freqs[mask], psd[mask], linewidth=1.5, alpha=0.7, color=colors[j])
            else:
                # Linear scale
                ax.plot(freqs[mask], psd[mask], linewidth=1.5, alpha=0.7, color=colors[j])

        # Add title and labels
        ax.set_title(f'{title_prefix} {i+1}')
        ax.set_xlabel('Frequency (Hz)')
        scale_text = "Log " if log_scale else ""
        ax.set_ylabel(f'{scale_text}Power')
        ax.grid(True, which="both" if log_scale else "major")

        # Add minor grid lines for log scale
        if log_scale:
            ax.grid(True, which="minor", axis="y", linestyle=":", alpha=0.4)

        # Add colored backgrounds for frequency bands
        y_min, y_max = ax.get_ylim()
        for (low, high), color, name in zip(band_ranges, band_colors, band_names):
            if high >= 5:  # Only show bands that overlap with our display range
                if i == 0:  # Only add labels to the first plot
                    ax.axvspan(max(5, low), min(high, 70), alpha=0.1, color=color,
                              label=name if i == 0 else "")
                else:
                    ax.axvspan(max(5, low), min(high, 70), alpha=0.1, color=color)

    # Add a single legend for channels at the bottom of the figure
    handles = [plt.Line2D([0], [0], color=colors[j], lw=2, alpha=0.7, label=f'Ch {j}')
              for j in range(min(num_channels, segment.shape[0]))]
    plt.figlegend(handles=handles, loc='lower center', ncol=min(num_channels, 6),
                  bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Make room for legend
    plt.savefig(output_path)
    print(f"Multi-segment power spectrum plots saved to {output_path}")
    plt.close()

def plot_multi_segment_mutual_info(segments, num_channels, title_prefix="Segment", output_path="plot.png"):
    """
    Create a single figure with mutual information matrices for multiple segments.
    
    Args:
        segments: List of segmented EEG data
        num_channels: Number of channels to analyze
        title_prefix: Prefix for subplot titles
        output_path: Path to save the plot
    """
    num_segments = len(segments)
    if num_segments == 0:
        return
    
    # Calculate the grid dimensions
    grid_size = int(np.ceil(np.sqrt(num_segments)))
    rows = int(np.ceil(num_segments / grid_size))
    cols = grid_size
    
    # Create figure
    plt.figure(figsize=(cols*5, rows*4))
    
    # Create a subplot for each segment
    for i, segment in enumerate(segments):
        # Limit to the specified number of channels
        if num_channels is None:
            seg_num_channels = segment.shape[0]
        else:
            seg_num_channels = min(num_channels, segment.shape[0])
            
        # Select channels to analyze
        data_to_analyze = segment[:seg_num_channels]
        
        # Calculate mutual information matrix
        print(f"Calculating MI matrix for segment {i+1}/{num_segments}...")
        mi_matrix = calculate_mutual_information_matrix(data_to_analyze)
        
        # Create subplot
        ax = plt.subplot(rows, cols, i+1)
        
        # Create heatmap
        im = ax.imshow(mi_matrix, cmap=plt.cm.viridis, vmin=0, vmax=1)
        
        # Add title
        ax.set_title(f'{title_prefix} {i+1}')
        
        # Only add labels to subplots on the left and bottom edges
        if i % cols == 0:  # Left edge
            ax.set_ylabel('Channel')
        if i >= (rows-1) * cols:  # Bottom edge
            ax.set_xlabel('Channel')
        
        # Simplify ticks for readability
        channel_step = max(1, seg_num_channels // 5)  # Show at most 5 ticks
        ax.set_xticks(np.arange(0, seg_num_channels, channel_step))
        ax.set_yticks(np.arange(0, seg_num_channels, channel_step))
        ax.set_xticklabels(np.arange(0, seg_num_channels, channel_step))
        ax.set_yticklabels(np.arange(0, seg_num_channels, channel_step))
    
    # Add a colorbar to the right of the figure
    cbar_ax = plt.axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    plt.colorbar(im, cax=cbar_ax, label='Mutual Information')
    
    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Make room for colorbar
    plt.savefig(output_path)
    print(f"Multi-segment mutual information matrix plot saved to {output_path}")
    plt.close()

def plot_multi_segment_corr(segments, num_channels, title_prefix="Segment", output_path="plot.png"):
    """
    Create a single figure with correlation matrices for multiple segments.

    Args:
        segments: List of segmented EEG data
        num_channels: Number of channels to analyze
        title_prefix: Prefix for subplot titles
        output_path: Path to save the plot
    """
    num_segments = len(segments)
    if num_segments == 0:
        return

    # Calculate the grid dimensions
    grid_size = int(np.ceil(np.sqrt(num_segments)))
    rows = int(np.ceil(num_segments / grid_size))
    cols = grid_size

    # Create figure
    plt.figure(figsize=(cols*5, rows*4))

    # Create a subplot for each segment
    for i, segment in enumerate(segments):
        # Limit to the specified number of channels
        if num_channels is None:
            seg_num_channels = segment.shape[0]
        else:
            seg_num_channels = min(num_channels, segment.shape[0])

        # Select channels to analyze
        data_to_analyze = segment[:seg_num_channels]

        # Calculate correlation matrix
        corr_matrix = np.corrcoef(data_to_analyze)

        # Create subplot
        ax = plt.subplot(rows, cols, i+1)

        # Create heatmap
        im = ax.imshow(corr_matrix, cmap=plt.cm.RdBu_r, vmin=-1, vmax=1)

        # Add title
        ax.set_title(f'{title_prefix} {i+1}')

        # Only add labels to subplots on the left and bottom edges
        if i % cols == 0:  # Left edge
            ax.set_ylabel('Channel')
        if i >= (rows-1) * cols:  # Bottom edge
            ax.set_xlabel('Channel')

        # Simplify ticks for readability
        channel_step = max(1, seg_num_channels // 5)  # Show at most 5 ticks
        ax.set_xticks(np.arange(0, seg_num_channels, channel_step))
        ax.set_yticks(np.arange(0, seg_num_channels, channel_step))
        ax.set_xticklabels(np.arange(0, seg_num_channels, channel_step))
        ax.set_yticklabels(np.arange(0, seg_num_channels, channel_step))

    # Add a colorbar to the right of the figure
    cbar_ax = plt.axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    plt.colorbar(im, cax=cbar_ax, label='Correlation')

    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Make room for colorbar
    plt.savefig(output_path)
    print(f"Multi-segment correlation matrix plot saved to {output_path}")
    plt.close()

def run_segmented_analysis(eeg_data, analysis_types, num_segments, num_channels, output_path="plot.png", normalize=False):
    """
    Divide data into segments and create a single figure with multiple segment analyses.

    Args:
        eeg_data: EEG data with shape [channels, frames]
        analysis_types: List of analysis types (['std', 'qq', 'corr'])
        num_segments: Number of segments to divide the data into
        num_channels: Number of channels to analyze
        output_path: Path to save the plot
        normalize: Whether to normalize data for QQ plots

    Returns:
        Output path of the combined plot
    """
    if eeg_data is None or num_segments < 1 or not analysis_types:
        print("No data to analyze, invalid segment count, or no analysis types specified")
        return None

    # Segment the data
    segments = segment_data(eeg_data, num_segments)

    # For a single analysis type, use the original functions
    if len(analysis_types) == 1:
        analysis_type = analysis_types[0]
        if analysis_type == 'std':
            plot_multi_segment_std(segments, num_channels, output_path=output_path)
        elif analysis_type == 'qq':
            plot_multi_segment_qq(segments, num_channels, output_path=output_path)
        elif analysis_type == 'corr':
            plot_multi_segment_corr(segments, num_channels, output_path=output_path)
        elif analysis_type == 'mi':
            plot_multi_segment_mutual_info(segments, num_channels, output_path=output_path)
        elif analysis_type == 'psd':
            plot_multi_segment_psd(segments, num_channels, output_path=output_path,
                                log_scale=args.log_scale, apply_filter=args.apply_filter,
                                low_freq=args.low_freq, high_freq=args.high_freq)
        elif analysis_type == 'fooof':
            if FOOOF_AVAILABLE:
                plot_multi_segment_fooof(segments, num_channels, output_path=output_path,
                                    log_scale=args.log_scale, apply_filter=args.apply_filter,
                                    low_freq=max(3.0, args.low_freq), high_freq=args.high_freq)
            else:
                print("FOOOF analysis requested but FOOOF package not available.")
                print("Please install with: pip install fooof")
        elif analysis_type == 'flat':
            plot_multi_segment_flatlines(segments, num_channels, output_path=output_path)
        elif analysis_type == 'range':
            plot_multi_segment_range(segments, num_channels, output_path=output_path)
        else:
            print(f"Unsupported analysis type: {analysis_type}")
            return None
        return output_path

    # For multiple analysis types, create a combined figure
    return plot_multi_analysis(segments, analysis_types, num_channels, output_path, normalize)

def plot_multi_analysis(segments, analysis_types, num_channels, output_path="plot.png", normalize=False):
    """
    Create a single figure with multiple analysis types for multiple segments.

    Args:
        segments: List of segmented EEG data
        analysis_types: List of analysis types to perform
        num_channels: Number of channels to analyze
        output_path: Path to save the plot
        normalize: Whether to normalize data for QQ plots

    Returns:
        Output path of the combined plot
    """
    num_segments = len(segments)
    num_analyses = len(analysis_types)

    if num_segments == 0 or num_analyses == 0:
        return None

    # Create a grid with analyses as rows and segments as columns
    fig = plt.figure(figsize=(num_segments * 4, num_analyses * 4))

    # Set up a consistent colormap for channels
    colors = plt.cm.viridis(np.linspace(0, 1, num_channels))

    # Create a grid of subplots
    gs = plt.GridSpec(num_analyses, num_segments, figure=fig)

    # Process each analysis type and segment
    for a_idx, analysis_type in enumerate(analysis_types):
        for s_idx, segment in enumerate(segments):
            # Create a subplot for this analysis and segment
            ax = fig.add_subplot(gs[a_idx, s_idx])

            # Set title based on position
            if a_idx == 0:  # Top row
                ax.set_title(f"Segment {s_idx+1}")

            if s_idx == 0:  # First column
                ax.set_ylabel(analysis_type.upper())

            # Perform the specific analysis
            if analysis_type == 'std':
                # Standard deviation analysis
                std_values = analyze_channel_std(segment, num_channels)
                channel_indices = np.arange(len(std_values))
                ax.bar(channel_indices, std_values, color='steelblue')
                ax.set_xlabel('Channel')
                ax.grid(axis='y', linestyle='--', alpha=0.7)

            elif analysis_type == 'qq':
                # Q-Q plot analysis
                for j in range(min(num_channels, segment.shape[0])):
                    channel_data = segment[j].copy()

                    # Normalize data if requested
                    if normalize:
                        channel_data = normalize_channel(channel_data)

                    # Get theoretical quantiles and ordered data
                    osm = stats.probplot(channel_data, dist="norm", fit=False)
                    theoretical_quantiles = osm[0]
                    ordered_data = osm[1]

                    # Plot with color for this channel
                    ax.plot(theoretical_quantiles, ordered_data, 'o',
                           markersize=2, alpha=0.5, color=colors[j])

                # Add reference line
                if normalize:
                    # For normalized data, reference line should be y=x
                    ax.plot([-3, 3], [-3, 3], 'k--')
                else:
                    # For unnormalized data, determine appropriate reference line
                    min_x = min([min(stats.probplot(segment[j], dist="norm", fit=False)[0])
                                for j in range(min(num_channels, segment.shape[0]))])
                    max_x = max([max(stats.probplot(segment[j], dist="norm", fit=False)[0])
                                for j in range(min(num_channels, segment.shape[0]))])
                    ax.plot([min_x, max_x], [min_x, max_x], 'k--')

                ax.set_xlabel('Theoretical Quantiles')
                ax.grid(True)

            elif analysis_type == 'corr':
                # Correlation matrix analysis
                seg_num_channels = min(num_channels, segment.shape[0])
                data_to_analyze = segment[:seg_num_channels]
                corr_matrix = np.corrcoef(data_to_analyze)

                # Create heatmap
                im = ax.imshow(corr_matrix, cmap=plt.cm.RdBu_r, vmin=-1, vmax=1)

                # Simplify ticks
                channel_step = max(1, seg_num_channels // 5)
                ax.set_xticks(np.arange(0, seg_num_channels, channel_step))
                ax.set_yticks(np.arange(0, seg_num_channels, channel_step))
                ax.set_xticklabels(np.arange(0, seg_num_channels, channel_step))
                ax.set_yticklabels(np.arange(0, seg_num_channels, channel_step))

                # Add colorbar for the first correlation plot only
                if a_idx == analysis_types.index('corr') and s_idx == 0:
                    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
                    fig.colorbar(im, cax=cbar_ax, label='Correlation')
                    
            elif analysis_type == 'mi':
                # Mutual information matrix analysis
                seg_num_channels = min(num_channels, segment.shape[0])
                data_to_analyze = segment[:seg_num_channels]
                
                # Calculate mutual information matrix
                print(f"Calculating MI matrix for segment {s_idx+1}...")
                mi_matrix = calculate_mutual_information_matrix(data_to_analyze)
                
                # Create heatmap
                im = ax.imshow(mi_matrix, cmap=plt.cm.viridis, vmin=0, vmax=1)
                
                # Simplify ticks
                channel_step = max(1, seg_num_channels // 5)
                ax.set_xticks(np.arange(0, seg_num_channels, channel_step))
                ax.set_yticks(np.arange(0, seg_num_channels, channel_step))
                ax.set_xticklabels(np.arange(0, seg_num_channels, channel_step))
                ax.set_yticklabels(np.arange(0, seg_num_channels, channel_step))
                
                # Add colorbar for the first mutual information plot only
                if 'mi' in analysis_types and a_idx == analysis_types.index('mi') and s_idx == 0:
                    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
                    fig.colorbar(im, cax=cbar_ax, label='Mutual Information')

            elif analysis_type == 'psd':
                # Power spectrum analysis

                # Common EEG frequency bands
                band_colors = ['#e6194B', '#f58231', '#ffe119', '#3cb44b', '#4363d8']
                band_names = ['Theta', 'Alpha', 'Beta', 'Low Gamma', 'High Gamma']
                band_ranges = [(4, 8), (8, 13), (13, 30), (30, 50), (50, 70)]

                # Use log scale for PSD and apply filtering
                log_scale = True  # Hard-coded to true for multi-analysis view
                apply_filter = True  # Hard-coded to true for multi-analysis view
                low_freq = 2.0
                high_freq = 45.0

                # Apply bandpass filter
                filtered_segment = segment.copy()
                if apply_filter:
                    filtered_segment = apply_bandpass_filter(filtered_segment, 500, low_freq, high_freq)
                    filter_text = f" (Filtered: {low_freq}-{high_freq} Hz)"
                else:
                    filter_text = ""

                # Calculate and plot power spectrum for each channel
                for j in range(min(num_channels, segment.shape[0])):
                    channel_data = filtered_segment[j]

                    # Use Welch's method to calculate power spectral density
                    freqs, psd = signal.welch(channel_data, 500, nperseg=min(2048, segment.shape[1]))

                    # Plot frequency range based on filter settings
                    upper_freq = min(70, high_freq + 10)  # Show a bit above the filter cutoff
                    mask = (freqs >= max(1, low_freq - 1)) & (freqs <= upper_freq)

                    if log_scale:
                        # Use semilogy for log scale on y-axis
                        ax.semilogy(freqs[mask], psd[mask], linewidth=1.5, alpha=0.7, color=colors[j])
                    else:
                        # Linear scale
                        ax.plot(freqs[mask], psd[mask], linewidth=1.5, alpha=0.7, color=colors[j])

                # Add colored backgrounds for frequency bands
                y_min, y_max = ax.get_ylim()
                for (low, high), color, name in zip(band_ranges, band_colors, band_names):
                    # Only show bands that overlap with our display range and filter range
                    if high >= max(1, low_freq - 1) and low <= upper_freq:
                        ax.axvspan(max(max(1, low_freq - 1), low), min(high, upper_freq), alpha=0.1, color=color)

                ax.set_xlabel('Frequency (Hz)')
                scale_text = "Log " if log_scale else ""
                ax.set_ylabel(f'{scale_text}Power')
                ax.grid(True, which="both" if log_scale else "major")

                # Add minor grid lines for log scale
                if log_scale:
                    ax.grid(True, which="minor", axis="y", linestyle=":", alpha=0.4)
                    
            elif analysis_type == 'fooof' and FOOOF_AVAILABLE:
                # FOOOF power spectrum decomposition analysis
                
                # Colors for different components
                component_colors = {
                    'original': 'black',
                    'model': 'red',
                    'aperiodic': 'blue',
                    'periodic': 'green'
                }
                
                # EEG frequency bands (for visualization)
                band_colors = ['#e6194B', '#f58231', '#ffe119', '#3cb44b']
                band_names = ['Theta', 'Alpha', 'Beta', 'Gamma']
                band_ranges = [(4, 8), (8, 13), (13, 30), (30, 45)]
                
                # Use log scale for PSD and apply filtering
                log_scale = True  # Hard-coded to true for multi-analysis view
                apply_filter = True  # Hard-coded to true for multi-analysis view
                low_freq = 3.0
                high_freq = 45.0
                
                # Apply bandpass filter
                filtered_segment = segment.copy()
                if apply_filter:
                    filtered_segment = apply_bandpass_filter(filtered_segment, 500, low_freq, high_freq)
                
                # Use data from the first channel only to keep the plot clean
                channel_data = filtered_segment[0]
                
                # Use Welch's method to calculate power spectral density
                freqs, psd = signal.welch(channel_data, 500, nperseg=min(2048, segment.shape[1]))
                
                # Restrict to frequency range of interest
                mask = (freqs >= low_freq) & (freqs <= high_freq)
                freqs_masked = freqs[mask]
                psd_masked = psd[mask]
                
                # Analyze with FOOOF
                fooof_results = analyze_fooof(freqs_masked, psd_masked, freq_range=(low_freq, high_freq))
                
                # Plot the components
                if log_scale:
                    ax.semilogy(freqs_masked, psd_masked, color=component_colors['original'], linewidth=1.5, alpha=0.7, label='Original')
                    ax.semilogy(freqs_masked, fooof_results['aperiodic'], color=component_colors['aperiodic'], linewidth=1.5, alpha=0.7, linestyle='--', label='Aperiodic')
                    ax.semilogy(freqs_masked, fooof_results['periodic'], color=component_colors['periodic'], linewidth=1.5, alpha=0.7, label='Periodic')
                else:
                    ax.plot(freqs_masked, psd_masked, color=component_colors['original'], linewidth=1.5, alpha=0.7, label='Original')
                    ax.plot(freqs_masked, fooof_results['aperiodic'], color=component_colors['aperiodic'], linewidth=1.5, alpha=0.7, linestyle='--', label='Aperiodic')
                    ax.plot(freqs_masked, fooof_results['periodic'], color=component_colors['periodic'], linewidth=1.5, alpha=0.7, label='Periodic')
                
                # Mark peaks
                for peak in fooof_results['peaks']:
                    if low_freq <= peak[0] <= high_freq:
                        peak_freq, peak_power, peak_width = peak
                        if log_scale:
                            ax.plot(peak_freq, 10**peak_power, 'o', markersize=5, 
                                   alpha=0.7, color=component_colors['periodic'])
                        else:
                            ax.plot(peak_freq, peak_power, 'o', markersize=5, 
                                   alpha=0.7, color=component_colors['periodic'])
                
                ax.set_xlabel('Frequency (Hz)')
                ax.set_ylabel('Power')
                ax.grid(True, which="both" if log_scale else "major")
                
                if s_idx == 0:  # Only add legend to first segment
                    ax.legend(fontsize=8, loc='upper right')
                
                # Add R-squared value to title
                r_squared = fooof_results.get('r_squared', 0)
                ax.set_title(f'FOOOF (R²={r_squared:.2f})')

            elif analysis_type == 'flat':
                # Flatline analysis

                # Need a minimum amount of data
                window_size = min(2500, segment.shape[1] // 2)  # Adjust window size if segment is small
                if window_size < 100:  # Too small for meaningful analysis
                    ax.text(0.5, 0.5, 'Insufficient data for flatline analysis',
                           horizontalalignment='center', verticalalignment='center',
                           transform=ax.transAxes)
                    continue

                # Limit to specified number of channels
                seg_num_channels = min(num_channels, segment.shape[0])
                segment_data = segment[:seg_num_channels]

                # Calculate flatline percentages
                flatline_percentages = detect_flatlines(segment_data, window_size)

                if flatline_percentages is None:
                    ax.text(0.5, 0.5, 'Flatline analysis failed',
                           horizontalalignment='center', verticalalignment='center',
                           transform=ax.transAxes)
                    continue

                # Create bar chart
                channel_indices = np.arange(seg_num_channels)
                ax.bar(channel_indices, flatline_percentages, color='firebrick')

                # Add labels
                ax.set_xlabel('Channel')
                ax.set_ylabel('Flatline %')
                ax.grid(axis='y', linestyle='--', alpha=0.7)

                # Only show a subset of ticks for readability
                tick_step = max(1, seg_num_channels // 10)
                ax.set_xticks(channel_indices[::tick_step])

                # Add a horizontal line at 0%
                ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)

            elif analysis_type == 'range':
                # Sliding range analysis

                # Need a minimum amount of data for meaningful analysis
                window_size = min(2500, segment.shape[1] // 3)  # Adjust window size
                if segment.shape[1] < window_size * 2:  # Need at least 2 windows
                    ax.text(0.5, 0.5, 'Insufficient data for sliding range analysis',
                           horizontalalignment='center', verticalalignment='center',
                           transform=ax.transAxes)
                    continue

                # Limit to specified number of channels
                seg_num_channels = min(num_channels, segment.shape[0])
                segment_data = segment[:seg_num_channels]

                # Calculate sliding range
                step_size = max(window_size // 5, 1)  # Adjust step size for smaller segments
                time_points, channel_ranges = calculate_sliding_range(segment_data, window_size, step_size)

                if time_points is None or channel_ranges is None:
                    ax.text(0.5, 0.5, 'Sliding range analysis failed',
                           horizontalalignment='center', verticalalignment='center',
                           transform=ax.transAxes)
                    continue

                # Convert time points to seconds
                time_seconds = time_points / 500  # Assume 500Hz

                # Calculate median range across channels
                median_range = np.median(channel_ranges, axis=0)

                # Plot median range as a line
                ax.plot(time_seconds, median_range, 'b-', linewidth=1.5)

                # Add interquartile range as a shaded area
                q1_range = np.percentile(channel_ranges, 25, axis=0)  # 25th percentile (1st quartile)
                q3_range = np.percentile(channel_ranges, 75, axis=0)  # 75th percentile (3rd quartile)
                ax.fill_between(time_seconds, q1_range, q3_range, color='b', alpha=0.2)

                # Add labels
                ax.set_xlabel('Time (seconds)')
                ax.set_ylabel('Signal Range')
                ax.grid(True)

    # Add a legend for QQ plots
    if 'qq' in analysis_types:
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[j],
                             markersize=8, alpha=0.7, label=f'Ch {j}')
                  for j in range(min(num_channels, 6))]  # Show first 6 channels
        fig.legend(handles=handles, loc='lower center', ncol=min(num_channels, 6),
                  bbox_to_anchor=(0.5, 0.02))

    # Adjust layout
    if 'corr' in analysis_types:
        plt.tight_layout(rect=[0, 0.05, 0.9, 0.95])  # Make room for colorbar and legend
    else:
        plt.tight_layout(rect=[0, 0.05, 0.95, 0.95])  # Make room for legend only

    plt.savefig(output_path)
    print(f"Multi-analysis plot saved to {output_path}")
    plt.close()

    return output_path

def main():
    parser = argparse.ArgumentParser(
        description='Analyze EEG channel standard deviation',
        epilog="""
Example usage with preprocessing:
  python analyze_eeg.py --file data/session/eeg_file.h5 --analysis std,psd --transpose --clip-data --log-transform --reject-segments

This will:
  1. Load the EEG data from the specified file
  2. Swap dimensions 1 and 2 (channels and samples) during data loading if needed
  3. Apply amplitude clipping to remove extreme values (1st and 99th percentiles by default)
  4. Apply logarithmic transformation to amplitude values
  5. Divide the data into overlapping 5-second segments (with 0.5s overlap on each side)
  6. Calculate the median range (max-min) for each segment across channels
  7. Reject the 50% of segments with the highest ranges
  8. Rebuild a continuous signal using only the kept segments
  9. Run standard deviation and power spectrum analyses on the cleaned data

You can use preprocessing options independently or in combination:
  python analyze_eeg.py --file data/session/eeg_file.h5 --analysis std,psd --transpose
  python analyze_eeg.py --file data/session/eeg_file.h5 --analysis std,psd --clip-data --lower-quantile 0.05 --upper-quantile 0.95
  python analyze_eeg.py --file data/session/eeg_file.h5 --analysis std,psd --log-transform
  python analyze_eeg.py --file data/session/eeg_file.h5 --analysis std,psd --reject-segments

Channel selection:
  python analyze_eeg.py --file data/session/eeg_file.h5 --channels 16  # Use only first 16 channels
  python analyze_eeg.py --file data/session/eeg_file.h5 --exclude-channels "0,4,7"  # Exclude channels 0, 4, and 7
  
External data:
  python analyze_eeg.py --numpy-data "/path/to/data/*.npy" --analysis std,psd  # Load data from NumPy files
""")
    parser.add_argument('--file', type=str, help='Specific H5 file to analyze')
    parser.add_argument('--dir', type=str, default='./data',
                        help='Directory containing H5 files (default: ./data)')
    parser.add_argument('--output', type=str, default='plot.png',
                        help='Output plot file path (default: plot.png)')
    parser.add_argument('--channels', type=int, default=22,
                        help='Number of EEG channels to analyze (default: 22)')
    parser.add_argument('--exclude-channels', type=str, 
                        help='Comma-separated list of channel indices to exclude (e.g., "0,4,7")')
    parser.add_argument('--list-files', action='store_true',
                        help='Just list available H5 files without analysis')
    parser.add_argument('--analysis', type=str, default='std',
                        help='Analysis types separated by commas (std,qq,corr,mi,psd,flat,range) (default: std)')
    parser.add_argument('--segments', type=int, default=1,
                        help='Number of segments to divide the recording into (default: 1)')
    parser.add_argument('--normalize', action='store_true',
                        help='Normalize channel data (z-score) before analysis')
    parser.add_argument('--threshold', type=float, default=0.005,
                        help='Threshold for flatline detection (default: 0.005)')
    parser.add_argument('--log-scale', action='store_true', default=True,
                        help='Use logarithmic scale for power spectrum y-axis (default: True)')
    parser.add_argument('--no-filter', action='store_false', dest='apply_filter', default=True,
                        help='Disable bandpass filtering for power spectrum analysis')
    parser.add_argument('--low-freq', type=float, default=2.0,
                        help='Lower cutoff frequency in Hz for bandpass filter (default: 2.0)')
    parser.add_argument('--high-freq', type=float, default=45.0,
                        help='Upper cutoff frequency in Hz for bandpass filter (default: 45.0)')
    parser.add_argument('--window-size', type=float, default=5.0,
                        help='Window size in seconds for sliding window analyses (default: 5.0)')
    parser.add_argument('--reject-segments', action='store_true',
                        help='Enable preprocessing to reject segments with high amplitude range')
    parser.add_argument('--log-transform', action='store_true',
                        help='Apply logarithmic transformation to amplitude values before analysis')
    parser.add_argument('--transpose', action='store_true',
                        help='Swap dimensions 1 and 2 (channels and samples) during data loading (for formats like [chunks, samples, channels])')
    parser.add_argument('--numpy-data', type=str, 
                        help='Path to NumPy data files (supports glob patterns like "/path/to/*.npy")')
    parser.add_argument('--segment-duration', type=float, default=5.0,
                        help='Duration of segments in seconds for rejection preprocessing (default: 5.0)')
    parser.add_argument('--overlap', type=float, default=0.5,
                        help='Overlap in seconds between segments for rejection preprocessing (default: 0.5)')
    parser.add_argument('--reject-ratio', type=float, default=0.5,
                        help='Ratio of segments to reject based on highest range values (default: 0.5)')
    parser.add_argument('--bandpass-filter', action='store_true',
                        help='Apply 5-45 Hz bandpass filter to ALL data before any analysis')
    parser.add_argument('--clip-data', action='store_true',
                        help='Enable amplitude clipping based on quantiles')
    parser.add_argument('--lower-quantile', type=float, default=0.01, 
                        help='Lower quantile threshold for clipping (default: 0.01)')
    parser.add_argument('--upper-quantile', type=float, default=0.99,
                        help='Upper quantile threshold for clipping (default: 0.99)')
    args = parser.parse_args()

    # Find all H5 files in the specified directory
    h5_files = find_h5_files(args.dir)
    print(f"Found {len(h5_files)} H5 files in {args.dir}")

    if args.list_files:
        for i, file in enumerate(h5_files):
            print(f"{i+1}. {file}")
        return

    # Determine data source to analyze
    eeg_data = None
    
    if args.numpy_data:
        # Load data from NumPy files
        import glob
        numpy_files = glob.glob(args.numpy_data)
        if not numpy_files:
            print(f"Error: No NumPy files found matching pattern: {args.numpy_data}")
            return
            
        print(f"Found {len(numpy_files)} NumPy files matching {args.numpy_data}")
        # Load the first file for demonstration
        numpy_file = numpy_files[0]
        print(f"Loading NumPy data from {numpy_file}")
        eeg_data = np.load(numpy_file)
        print(f"Loaded NumPy data with shape: {eeg_data.shape}")
        
        # Handle exclude_channels if specified
        if args.exclude_channels:
            exclude_list = [int(ch) for ch in args.exclude_channels.split(',')]
            keep_indices = [i for i in range(eeg_data.shape[0]) if i not in exclude_list]
            eeg_data = eeg_data[keep_indices]
            print(f"Excluded channels: {args.exclude_channels}")
            print(f"New data shape: {eeg_data.shape}")
    else:
        # Use H5 files
        file_to_analyze = None
        if args.file:
            file_to_analyze = args.file
            print(f"Using specified file: {file_to_analyze}")

            # Check if file exists as specified
            if not os.path.isfile(file_to_analyze):
                # Try to find it in the directory
                basename = os.path.basename(file_to_analyze)
                matching_files = [f for f in h5_files if basename in f]
                if matching_files:
                    file_to_analyze = matching_files[0]
                    print(f"Found matching file: {file_to_analyze}")
                else:
                    print(f"Error: File not found: {file_to_analyze}")
                    return
        elif h5_files:
            file_to_analyze = h5_files[0]
            print(f"No file specified. Using first H5 file: {file_to_analyze}")
        else:
            print("No H5 files found to analyze")
            return

        # Load the data, with transpose if requested
        print(f"Loading data with transpose={args.transpose}")
        eeg_data = load_eeg_data(file_to_analyze, transpose=args.transpose, exclude_channels=args.exclude_channels)

    if eeg_data is None:
        print("Failed to load EEG data, cannot proceed with analysis")
        return
        
    # Apply bandpass filter if requested (before any other preprocessing)
    if args.bandpass_filter:
        print("Applying 5-45 Hz bandpass filter to all data...")
        eeg_data = apply_bandpass_filter(eeg_data, sample_rate=500, low_freq=5.0, high_freq=45.0)
        
    # Apply amplitude clipping if requested (after filtering but before log transform)
    if args.clip_data:
        print("Applying amplitude clipping based on quantiles...")
        eeg_data = clip_data(eeg_data, lower_quantile=args.lower_quantile, upper_quantile=args.upper_quantile)

    # Apply log transform preprocessing if requested (after potential filtering and clipping)
    if args.log_transform:
        print("Applying logarithmic transformation to amplitude values...")
        eeg_data = preprocess_log_transform(eeg_data)

    # Apply segment rejection preprocessing if requested
    if args.reject_segments:
        print("Applying segment rejection preprocessing...")
        # Create a path for the rejection visualization plot
        rejection_plot_path = args.output.replace('.png', '_segment_rejection.png')
        eeg_data = preprocess_reject_segments(
            eeg_data,
            segment_duration=args.segment_duration,
            overlap=args.overlap,
            reject_ratio=args.reject_ratio,
            plot_path=rejection_plot_path
        )

    # Parse analysis types from the comma-separated string
    analysis_types = [analysis_type.strip() for analysis_type in args.analysis.split(',')]

    # Validate analysis types
    valid_types = {'std', 'qq', 'corr', 'psd', 'flat', 'range', 'mi', 'fooof'}
    invalid_types = [t for t in analysis_types if t not in valid_types]
    if invalid_types:
        print(f"Warning: Ignoring unknown analysis types: {', '.join(invalid_types)}")
        analysis_types = [t for t in analysis_types if t in valid_types]

    if not analysis_types:
        print("No valid analysis types specified, defaulting to 'std'")
        analysis_types = ['std']

    # Determine whether to run segmented analysis or a single analysis
    if args.segments > 1:
        print(f"Dividing recording into {args.segments} segments for analysis")
        print(f"Analysis types: {', '.join(analysis_types)}")

        # Create a combined plot with all segments and analyses
        output_path = run_segmented_analysis(
            eeg_data,
            analysis_types,
            args.segments,
            args.channels,
            output_path=args.output,
            normalize=args.normalize
        )

        print(f"Created multi-segment and multi-analysis plot at {output_path}")

    else:
        # Single segment analysis
        if len(analysis_types) > 1:
            # Multiple analysis types on a single segment - use the multi-analysis plotting
            segments = [eeg_data]  # Create a single-element list
            output_path = plot_multi_analysis(segments, analysis_types, args.channels, args.output, normalize=args.normalize)
            print(f"Created multi-analysis plot at {output_path}")
        else:
            # Standard single analysis (no segmentation)
            analysis_type = analysis_types[0]

            if analysis_type == 'std':
                # Standard deviation analysis
                std_per_channel = analyze_channel_std(eeg_data, args.channels)

                # Display results
                if std_per_channel is not None:
                    print("\nChannel Standard Deviations:")
                    for i, std in enumerate(std_per_channel):
                        print(f"Channel {i}: {std:.4f}")

                    # Create plot
                    plot_channel_std(std_per_channel, args.output)

            elif analysis_type == 'qq':
                # Q-Q plot analysis for checking normality
                num_channels_to_plot = min(6, args.channels)  # Default to 6 channels or fewer if requested
                plot_qq_channels(eeg_data, num_channels_to_plot, args.output, normalize=args.normalize)
                norm_text = "normalized " if args.normalize else ""
                print(f"Created Q-Q plot for the first {num_channels_to_plot} {norm_text}channels")

            elif analysis_type == 'corr':
                # Correlation matrix analysis
                plot_correlation_matrix(eeg_data, args.channels, args.output)
                print(f"Created correlation matrix for {min(args.channels, eeg_data.shape[0])} channels")
                
            elif analysis_type == 'mi':
                # Mutual information matrix analysis
                plot_mutual_information_matrix(eeg_data, args.channels, args.output)
                print(f"Created mutual information matrix for {min(args.channels, eeg_data.shape[0])} channels")

            elif analysis_type == 'psd':
                # Power spectrum analysis
                num_channels_to_plot = min(6, args.channels)
                plot_power_spectrum(eeg_data, num_channels_to_plot, output_path=args.output,
                                  log_scale=args.log_scale, apply_filter=args.apply_filter,
                                  low_freq=args.low_freq, high_freq=args.high_freq)

                scale_text = "logarithmic" if args.log_scale else "linear"
                filter_text = f", filtered {args.low_freq}-{args.high_freq} Hz" if args.apply_filter else ", unfiltered"
                print(f"Created power spectrum plot ({scale_text} y-axis{filter_text}) for the first {num_channels_to_plot} channels")

            elif analysis_type == 'fooof':
                # FOOOF power spectrum decomposition analysis
                if FOOOF_AVAILABLE:
                    num_channels_to_plot = min(6, args.channels)
                    fooof_low_freq = max(3.0, args.low_freq)  # FOOOF needs at least 3Hz for stable fitting
                    plot_fooof_spectrum(eeg_data, num_channels_to_plot, output_path=args.output,
                                     log_scale=args.log_scale, apply_filter=args.apply_filter,
                                     low_freq=fooof_low_freq, high_freq=args.high_freq)
                    
                    scale_text = "logarithmic" if args.log_scale else "linear"
                    filter_text = f", filtered {fooof_low_freq}-{args.high_freq} Hz" if args.apply_filter else ", unfiltered"
                    print(f"Created FOOOF spectrum analysis ({scale_text} y-axis{filter_text}) for the first {num_channels_to_plot} channels")
                else:
                    print("FOOOF analysis requested but FOOOF package not available.")
                    print("Please install with: pip install fooof")

            elif analysis_type == 'flat':
                # Flatline analysis
                plot_flatlines(eeg_data, args.channels, threshold=args.threshold, output_path=args.output)
                print(f"Created flatline analysis plot for {min(args.channels, eeg_data.shape[0])} channels (threshold: ±{args.threshold})")

            elif analysis_type == 'range':
                # Sliding range analysis
                window_samples = int(args.window_size * 500)  # Convert seconds to samples (assuming 500Hz)
                plot_sliding_range(eeg_data, args.channels, window_size=window_samples, output_path=args.output)
                print(f"Created sliding range plot for {min(args.channels, eeg_data.shape[0])} channels (window: {args.window_size}s)")

if __name__ == "__main__":
    main()
