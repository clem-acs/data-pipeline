"""
Functions for preprocessing EEG data before windowing.
"""

import numpy as np
from scipy import signal


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
        # TEMPORARY FIX: currenly not saving to metadata
        sample_rate = 500
        # raise ValueError("Sample rate must be provided in metadata for EEG preprocessing")
    else:
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

    # Reshape the filtered data back to a 3D array using the original samples_per_frame
    # The filtered_data is currently (total_samples, channels)
    # We need to reshape it to match the fNIRS dimension ordering: (frames, channels, samples_per_frame)
    # This ensures consistent dimension ordering between modalities

    # Check that we have the right number of samples
    if total_samples != filtered_data.shape[0]:
        raise ValueError(f"Sample count mismatch after filtering: {filtered_data.shape[0]} vs expected {total_samples}")

    # First reshape to (frames, samples_per_frame, channels)
    intermediate_shape = filtered_data.reshape(frames, samples_per_frame, channels)
    
    # Then transpose to (frames, channels, samples_per_frame) for consistency with fNIRS
    preprocessed_eeg_data = np.transpose(intermediate_shape, (0, 2, 1))

    # Include sample rate and filter details in metadata
    preprocessing_metadata = {
        'bandpass_low_hz': bandpass_low,
        'bandpass_high_hz': bandpass_high,
        'bandpass_order': filter_order,
        'filters_applied': ['bandpass'],
        'channels_used': channels,
        'sample_rate': sample_rate
    }

    return preprocessed_eeg_data, preprocessing_metadata
