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
    # This is a placeholder implementation - replace with actual EEG preprocessing logic
    preprocessed_eeg_data = eeg_data.copy()
    
    # Example metadata to track
    preprocessing_metadata = {
        'bandpass_low_hz': 0.5,
        'bandpass_high_hz': 40.0,
        'filters_applied': ['bandpass'],
    }
    
    return preprocessed_eeg_data, preprocessing_metadata