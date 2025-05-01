"""
Functions for preprocessing EEG data before windowing.
"""

import numpy as np
from scipy import signal


def expand_eeg_timestamps(timestamps, metadata):
    """
    Expand EEG timestamps to match flattened EEG data.
    
    Args:
        timestamps: Original timestamps array with shape (frames, 3)
        metadata: Dictionary containing metadata about the signal, should include:
                  - sample_rate: Sampling rate in Hz
                  - frame_size: (optional) Number of samples per frame
    
    Returns:
        Tuple of (expanded_timestamps, timestamp_metadata)
    """
    # Check that sample rate is available in metadata
    if 'sample_rate' not in metadata:
        raise ValueError("Sample rate must be provided in metadata for timestamp expansion")
    sample_rate = metadata['sample_rate']
    
    # Get frames and samples_per_frame
    frames = timestamps.shape[0]
    
    # Get samples_per_frame from metadata or use default value (15)
    samples_per_frame = metadata.get('frame_size', 15)
    
    # Calculate time between samples in milliseconds
    sample_interval_ms = 1000.0 / sample_rate
    
    # Create expanded timestamps array (we only keep the device timestamp column)
    total_samples = frames * samples_per_frame
    expanded_timestamps = np.zeros((total_samples, 1), dtype=timestamps.dtype)
    
    # For each original timestamp, create interpolated timestamps
    for frame_idx in range(frames):
        # Get the device timestamp (first column)
        device_ts = timestamps[frame_idx, 0]
        
        # Create interpolated timestamps for each sample in the frame
        for sample_idx in range(samples_per_frame):
            # Calculate time offset for this sample in milliseconds
            sample_offset_ms = sample_idx * sample_interval_ms
            
            # Calculate new device timestamp
            new_device_ts = device_ts + sample_offset_ms
            
            # Store in expanded array
            expanded_idx = frame_idx * samples_per_frame + sample_idx
            expanded_timestamps[expanded_idx, 0] = new_device_ts
    
    # Create metadata for the timestamp processing
    timestamp_metadata = {
        'timestamp_source': 'device_timestamp',
        'original_timestamp_shape': timestamps.shape,
        'expanded_timestamp_shape': expanded_timestamps.shape,
        'sample_rate': sample_rate,
        'samples_per_frame': samples_per_frame,
        'interpolation_method': 'linear' 
    }
    
    return expanded_timestamps, timestamp_metadata


def preprocess_eeg(eeg_data, metadata):
    """
    Preprocess EEG data before windowing (bandpass filtering, etc).
    
    Args:
        eeg_data: EEG data array
        metadata: Dictionary of metadata about the signal
    
    Returns:
        Tuple of (preprocessed_eeg_data, preprocessing_metadata)
    """
    # TEMPORARY FIX: We're only using the first 21 channels which are the main EEG channels
    # and discarding channels 22-28 which are auxiliary channels
    eeg_data_trimmed = eeg_data[:, :21, :]
    
    # Check that sample rate is available in metadata - required for proper processing
    if 'sample_rate' not in metadata:
        raise ValueError("Sample rate must be provided in metadata for EEG preprocessing")
    sample_rate = metadata['sample_rate']
    
    # Reshape to 2D by flattening frames and samples
    # Each frame contains 15 samples, so we'll reshape to put all samples on the same dimension
    frames, channels, samples_per_frame = eeg_data_trimmed.shape
    
    # Rearrange data to make samples the primary dimension
    # Need to transpose to get the correct ordering when reshaping
    eeg_data_transposed = np.transpose(eeg_data_trimmed, (0, 2, 1))  # (frames, samples_per_frame, channels)
    total_samples = frames * samples_per_frame
    eeg_data_2d = eeg_data_transposed.reshape(total_samples, channels)
    
    # Apply bandpass filter to each channel
    # Define bandpass parameters
    bandpass_low = 0.5  # Lower cutoff frequency (Hz)
    bandpass_high = 40.0  # Upper cutoff frequency (Hz)
    filter_order = 4  # Filter order (4 is typically a good balance)
    
    # Create filter 
    nyquist = 0.5 * sample_rate
    low = bandpass_low / nyquist
    high = bandpass_high / nyquist
    b, a = signal.butter(filter_order, [low, high], btype='band')
    
    # Apply filter to each channel
    filtered_data = np.zeros_like(eeg_data_2d)
    for i in range(channels):
        # Apply bandpass filter (using filtfilt for zero-phase filtering)
        filtered_data[:, i] = signal.filtfilt(b, a, eeg_data_2d[:, i])
    
    # Final preprocessed data
    preprocessed_eeg_data = filtered_data
    
    # Include sample rate and filter details in metadata
    preprocessing_metadata = {
        'bandpass_low_hz': bandpass_low,
        'bandpass_high_hz': bandpass_high,
        'bandpass_order': filter_order,
        'filters_applied': ['bandpass'],
        'channels_used': channels,
        'sample_rate': sample_rate,
    }
    
    return preprocessed_eeg_data, preprocessing_metadata