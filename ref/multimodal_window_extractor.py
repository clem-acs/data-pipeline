#!/usr/bin/env python3
"""
Multimodal Window Extractor: Specialized for extracting fNIRS+EEG windows from STROOP data.
Maintains native sampling rates for each modality and ensures proper time alignment.
Features GPU acceleration and efficient batch processing.
"""

import os
import numpy as np
import torch
import json
import logging
import argparse
import time
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import NeuroHDF5 handler from data module
from ..data.neuro_hdf5 import NeuroHDF5

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

def get_available_device(use_cpu=False):
    """
    Get the best available device for computation.

    Args:
        use_cpu: Force CPU usage even if GPU is available

    Returns:
        PyTorch device
    """
    if use_cpu:
        return torch.device("cpu")

    # Check for CUDA (NVIDIA GPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        return device

    # Check for MPS (Apple Silicon GPU)
    elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple Silicon GPU (MPS)")
        return device

    # Fallback to CPU
    else:
        device = torch.device("cpu")
        logger.info("Using CPU (no GPU available)")
        return device

def extract_windows_gpu(data, window_size, stride, device=None):
    """
    Extract windows from a recording using GPU acceleration.

    Args:
        data: Numpy array of shape [channels, timepoints]
        window_size: Size of each window
        stride: Stride between windows
        device: Device to use for acceleration

    Returns:
        Tensor of shape [num_windows, channels, window_size]
    """
    # Handle device
    if device is None:
        device = get_available_device()

    # Check if we're using MPS (Apple Silicon)
    is_mps = device.type == 'mps'

    # MPS doesn't support float64, so ensure we use float32
    if is_mps and data.dtype == np.float64:
        data = data.astype(np.float32)

    # Move data to appropriate device
    try:
        data_tensor = torch.tensor(data, device=device, dtype=torch.float32)
    except Exception as e:
        # Fall back to CPU if device transfer fails
        logger.warning(f"Failed to transfer data to {device}: {e}")
        logger.warning("Falling back to CPU for this operation")
        device = torch.device('cpu')
        data_tensor = torch.tensor(data, device=device, dtype=torch.float32)

    channels, timepoints = data_tensor.shape

    # Skip if data is too small for window
    if timepoints < window_size:
        logger.warning(f"Data length ({timepoints}) is smaller than window size ({window_size}). Skipping.")
        return torch.zeros((0, channels, window_size), device=device, dtype=torch.float32)

    # Calculate number of windows
    num_windows = max(0, (timepoints - window_size) // stride + 1)

    if num_windows <= 0:
        logger.warning(f"No windows could be extracted with window_size={window_size} and stride={stride}")
        return torch.zeros((0, channels, window_size), device=device, dtype=torch.float32)

    # For small number of windows or CPU, use the simple approach
    if num_windows <= 100 or device.type == 'cpu':
        # Pre-allocate output tensor
        windows = torch.zeros((num_windows, channels, window_size), device=device, dtype=torch.float32)

        # Extract windows in a loop
        for i in range(num_windows):
            start_idx = i * stride
            end_idx = start_idx + window_size
            windows[i] = data_tensor[:, start_idx:end_idx]
    else:
        # For larger datasets on GPU, use a more optimized approach
        try:
            # Use vectorized unfold operation (faster)
            windows = data_tensor.unfold(1, window_size, stride)
            # Reshape to [num_windows, channels, window_size]
            windows = windows.permute(1, 0, 2)
        except Exception as e:
            # Fall back to the simple approach if unfold fails
            logger.warning(f"Unfold operation failed: {e}. Using loop-based approach.")
            windows = torch.zeros((num_windows, channels, window_size), device=device, dtype=torch.float32)
            for i in range(num_windows):
                start_idx = i * stride
                end_idx = start_idx + window_size
                windows[i] = data_tensor[:, start_idx:end_idx]

    # For MPS, ensure we've done a sync before returning
    if is_mps:
        torch.mps.synchronize()

    return windows

def extract_multimodal_stroop_windows(neuro, rec_id, window_time_seconds, overlap_fraction, device=None):
    """
    Extract time-aligned multimodal windows from a STROOP recording.

    This function extracts windows based on a fixed time duration, not sample count,
    so EEG and fNIRS windows will have different numbers of samples based on their
    respective sampling rates.

    Args:
        neuro: NeuroHDF5 instance
        rec_id: Recording ID
        window_time_seconds: Window duration in seconds
        overlap_fraction: Overlap fraction between windows (0-1)
        device: Device to use for acceleration

    Returns:
        Tuple of (eeg_windows, fnirs_windows, metadata_list, subject_id)
    """
    # Get device if not provided
    if device is None:
        device = get_available_device()

    # Load recording metadata
    metadata = neuro.get_recording_metadata(rec_id)
    if not metadata:
        logger.warning(f"No metadata found for recording {rec_id}")
        return None, None, [], "unknown"

    # Extract essential info
    subject_id = metadata.get('subject_id', 'unknown')
    task_name = metadata.get('task_name', 'stroop')
    has_eeg = metadata.get('has_eeg', False)
    has_fnirs = metadata.get('has_fnirs', False)

    # Skip recordings without both modalities
    if not has_eeg or not has_fnirs:
        logger.info(f"Skipping {rec_id}: Missing {'EEG' if not has_eeg else 'fNIRS'} data")
        return None, None, [], subject_id

    # Load EEG data
    eeg_data = None
    eeg_windows = None
    eeg_sfreq = None

    try:
        # Use 'preprocessed' for EEG if available, otherwise 'raw'
        try:
            eeg_result = neuro.get_modality_data(rec_id, 'eeg', 'preprocessed')
        except:
            eeg_result = neuro.get_modality_data(rec_id, 'eeg', 'raw')

        eeg_data = eeg_result['data']
        eeg_sfreq = eeg_result['metadata'].get('sampling_rate', 250.0)
        eeg_ch_names = eeg_result['channels']

        logger.info(f"Loaded EEG data: {eeg_data.shape} channels at {eeg_sfreq}Hz")

        # Calculate window size and stride in samples for EEG
        eeg_window_samples = int(window_time_seconds * eeg_sfreq)
        eeg_stride_samples = int(eeg_window_samples * (1 - overlap_fraction))

        # Ensure data is float32 for GPU compatibility
        if eeg_data.dtype == np.float64:
            eeg_data = eeg_data.astype(np.float32)

        # Extract EEG windows
        eeg_windows = extract_windows_gpu(eeg_data, eeg_window_samples, eeg_stride_samples, device)

        # Move back to CPU
        eeg_windows = eeg_windows.cpu().numpy()
        logger.info(f"Extracted {eeg_windows.shape[0]} EEG windows with {eeg_windows.shape[1]} channels and {eeg_windows.shape[2]} samples each")

    except Exception as e:
        logger.error(f"Error extracting EEG data from {rec_id}: {e}")
        eeg_data = None
        eeg_windows = None

    # Load fNIRS data
    fnirs_data = None
    fnirs_windows = None
    fnirs_sfreq = None

    try:
        # fNIRS data is usually only available as 'raw'
        fnirs_result = neuro.get_modality_data(rec_id, 'fnirs', 'raw')
        fnirs_data = fnirs_result['data']
        fnirs_sfreq = fnirs_result['metadata'].get('sampling_rate', 10.0)
        fnirs_ch_names = fnirs_result['channels']

        logger.info(f"Loaded fNIRS data: {fnirs_data.shape} channels at {fnirs_sfreq}Hz")

        # Calculate window size and stride in samples for fNIRS
        fnirs_window_samples = int(window_time_seconds * fnirs_sfreq)
        fnirs_stride_samples = int(fnirs_window_samples * (1 - overlap_fraction))

        # Ensure data is float32
        if fnirs_data.dtype == np.float64:
            fnirs_data = fnirs_data.astype(np.float32)

        # Extract fNIRS windows
        fnirs_windows = extract_windows_gpu(fnirs_data, fnirs_window_samples, fnirs_stride_samples, device)

        # Move back to CPU
        fnirs_windows = fnirs_windows.cpu().numpy()
        logger.info(f"Extracted {fnirs_windows.shape[0]} fNIRS windows with {fnirs_windows.shape[1]} channels and {fnirs_windows.shape[2]} samples each")

    except Exception as e:
        logger.error(f"Error extracting fNIRS data from {rec_id}: {e}")
        fnirs_data = None
        fnirs_windows = None

    # If either modality failed, skip this recording
    if eeg_windows is None or fnirs_windows is None or eeg_windows.shape[0] == 0 or fnirs_windows.shape[0] == 0:
        logger.warning(f"Skipping {rec_id}: Failed to extract windows from one or both modalities")
        return None, None, [], subject_id

    # Create metadata for each window
    # We'll use the smaller number of windows as our total
    num_windows = min(eeg_windows.shape[0], fnirs_windows.shape[0])
    logger.info(f"Using {num_windows} aligned windows from both modalities")

    # Trim to same number of windows if needed
    if eeg_windows.shape[0] > num_windows:
        eeg_windows = eeg_windows[:num_windows]
    if fnirs_windows.shape[0] > num_windows:
        fnirs_windows = fnirs_windows[:num_windows]

    # Create metadata for each window
    metadata_list = []
    window_stride_time = window_time_seconds * (1 - overlap_fraction)

    # Get events if available
    events_data = None
    try:
        events_data = neuro.get_events(rec_id)
    except Exception as e:
        logger.warning(f"No events found for {rec_id}: {e}")

    for i in range(num_windows):
        # Calculate time indices
        start_time = i * window_stride_time
        end_time = start_time + window_time_seconds

        # Create window metadata
        window_metadata = {
            'recording_id': rec_id,
            'subject_id': subject_id,
            'task_name': task_name,
            'start_time': start_time,
            'end_time': end_time,
            'window_duration': window_time_seconds,
            'window_idx': i,
            'modalities': ['eeg', 'fnirs'],

            # EEG-specific info
            'eeg_ch_count': eeg_windows.shape[1],
            'eeg_sample_count': eeg_windows.shape[2],
            'eeg_sample_rate': eeg_sfreq,

            # fNIRS-specific info
            'fnirs_ch_count': fnirs_windows.shape[1],
            'fnirs_sample_count': fnirs_windows.shape[2],
            'fnirs_sample_rate': fnirs_sfreq,

            # Window time alignment
            'aligned_by': 'time',
            'window_stride_time': window_stride_time
        }

        # Extract events within this window if available
        if events_data and 'events' in events_data:
            # Calculate window boundaries in samples (EEG timebase)
            start_sample = int(start_time * eeg_sfreq)
            end_sample = int(end_time * eeg_sfreq)

            # Filter events within window
            window_events = []
            for event in events_data['events']:
                event_sample = event[0]
                if start_sample <= event_sample < end_sample:
                    # Convert to window-relative time
                    event_time = (event_sample - start_sample) / eeg_sfreq
                    event_code = event[2]

                    # Get description if available
                    event_desc = "unknown"
                    if events_data.get('descriptions') and str(event_code) in events_data['descriptions']:
                        event_desc = events_data['descriptions'][str(event_code)]

                    window_events.append({
                        'time': event_time,
                        'code': int(event_code),
                        'description': event_desc
                    })

            if window_events:
                window_metadata['events'] = window_events
                window_metadata['event_count'] = len(window_events)

        metadata_list.append(window_metadata)

    return eeg_windows, fnirs_windows, metadata_list, subject_id

def extract_stroop_windows_batched(hdf5_file, output_dir, window_time_seconds=2.0, window_overlap=0.5,
                                  windows_per_file=1000, num_workers=4, device=None, batch_size=5):
    """
    Extract time-aligned multimodal windows from STROOP recordings in batches.

    Args:
        hdf5_file: Path to NeuroHDF5 file
        output_dir: Directory to save extracted windows
        window_time_seconds: Window duration in seconds
        window_overlap: Overlap fraction between windows (0-1)
        windows_per_file: Number of windows to save per file
        num_workers: Number of worker threads
        device: Device to use for acceleration (None for auto-detection)
        batch_size: Number of recordings to process at once

    Returns:
        Dictionary with extraction statistics
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save extraction parameters
    params = {
        'hdf5_file': hdf5_file,
        'window_time_seconds': window_time_seconds,
        'window_overlap': window_overlap,
        'windows_per_file': windows_per_file,
        'extraction_date': time.strftime("%Y-%m-%d %H:%M:%S"),
        'gpu_accelerated': True,
        'multimodal_aligned': True
    }

    with open(os.path.join(output_dir, 'extraction_params.json'), 'w') as f:
        json.dump(params, f, indent=2)

    # Get recording IDs from file with both EEG and fNIRS data
    with NeuroHDF5(hdf5_file, 'r') as neuro:
        # Get all recording IDs
        all_recording_ids = neuro.get_recording_ids()

        # Filter recordings with "stroop" task that have both EEG and fNIRS data
        recording_ids = []
        for rec_id in all_recording_ids:
            try:
                metadata = neuro.get_recording_metadata(rec_id)
                has_eeg = metadata.get('has_eeg', False)
                has_fnirs = metadata.get('has_fnirs', False)
                task_name = metadata.get('task_name', '').lower()

                if has_eeg and has_fnirs and 'stroop' in task_name:
                    recording_ids.append(rec_id)
            except Exception as e:
                logger.warning(f"Skipping recording {rec_id}: {e}")

    total_recordings = len(recording_ids)
    logger.info(f"Found {total_recordings} STROOP recordings with both EEG and fNIRS data")

    if total_recordings == 0:
        logger.error("No suitable recordings found. Exiting.")
        return {
            'error': 'No suitable recordings found',
            'total_windows': 0
        }

    # Process recordings in batches
    all_windows = []
    all_metadata = []
    start_time = time.time()

    # Track statistics
    subject_counts = {}
    total_eeg_windows = 0
    total_fnirs_windows = 0
    total_aligned_windows = 0

    # By subject storage for better organization
    subject_data = {}

    # Process in batches
    for batch_start in range(0, total_recordings, batch_size):
        batch_end = min(batch_start + batch_size, total_recordings)
        batch_recordings = recording_ids[batch_start:batch_end]

        logger.info(f"Processing recordings {batch_start+1}-{batch_end} of {total_recordings}")

        with NeuroHDF5(hdf5_file, 'r') as neuro:
            for rec_id in tqdm(batch_recordings, desc="Extracting multimodal windows"):
                eeg_windows, fnirs_windows, metadata_list, subject_id = extract_multimodal_stroop_windows(
                    neuro, rec_id, window_time_seconds, window_overlap, device
                )

                # Skip if extraction failed
                if eeg_windows is None or fnirs_windows is None or len(metadata_list) == 0:
                    logger.warning(f"Skipping {rec_id}: Failed to extract windows")
                    continue

                # Update statistics
                if subject_id not in subject_counts:
                    subject_counts[subject_id] = 0
                subject_counts[subject_id] += len(metadata_list)

                total_eeg_windows += eeg_windows.shape[0]
                total_fnirs_windows += fnirs_windows.shape[0]
                total_aligned_windows += len(metadata_list)

                # Store by subject
                if subject_id not in subject_data:
                    subject_data[subject_id] = {
                        'eeg_windows': [],
                        'fnirs_windows': [],
                        'metadata': []
                    }

                subject_data[subject_id]['eeg_windows'].append(eeg_windows)
                subject_data[subject_id]['fnirs_windows'].append(fnirs_windows)
                subject_data[subject_id]['metadata'].extend(metadata_list)

    # Save windows by subject
    batch_count = 0
    for subject_id, data in tqdm(subject_data.items(), desc="Saving windows by subject"):
        # Concatenate all windows for this subject
        eeg_windows = np.concatenate(data['eeg_windows'], axis=0)
        fnirs_windows = np.concatenate(data['fnirs_windows'], axis=0)
        metadata = data['metadata']

        # Save in chunks
        num_windows = len(metadata)
        for chunk_start in range(0, num_windows, windows_per_file):
            chunk_end = min(chunk_start + windows_per_file, num_windows)

            # Get data slices
            eeg_chunk = eeg_windows[chunk_start:chunk_end]
            fnirs_chunk = fnirs_windows[chunk_start:chunk_end]
            metadata_chunk = metadata[chunk_start:chunk_end]

            # Create output filename
            output_file = os.path.join(output_dir, f"subject_{subject_id}_batch_{batch_count:04d}.npz")

            # Save to file
            np.savez_compressed(
                output_file,
                eeg=eeg_chunk,
                fnirs=fnirs_chunk,
                metadata=json.dumps(metadata_chunk),
                subject_id=subject_id
            )

            all_windows.append(output_file)
            all_metadata.extend(metadata_chunk)
            batch_count += 1

            logger.info(f"Saved {len(metadata_chunk)} multimodal windows for subject {subject_id} to {output_file}")

    # Calculate extraction time
    end_time = time.time()
    processing_time = end_time - start_time
    windows_per_second = total_aligned_windows / processing_time if processing_time > 0 else 0

    # Save index file
    index = {
        'num_batches': len(all_windows),
        'windows_per_batch': windows_per_file,
        'total_windows': total_aligned_windows,
        'batch_files': [os.path.basename(f) for f in all_windows],
        'subject_counts': subject_counts,
        'available_modalities': ['eeg', 'fnirs'],
        'modality_counts': {
            'eeg': total_eeg_windows,
            'fnirs': total_fnirs_windows,
            'aligned': total_aligned_windows
        },
        'extraction_params': params,
        'processing_time_seconds': processing_time,
        'windows_per_second': windows_per_second,
        'grouped_by_subject': True
    }

    with open(os.path.join(output_dir, 'index.json'), 'w') as f:
        json.dump(index, f, indent=2)

    # Print summary
    logger.info(f"Extraction complete!")
    logger.info(f"Total windows: {total_aligned_windows}")
    logger.info(f"Subject count: {len(subject_counts)}")
    logger.info(f"Batch count: {len(all_windows)}")
    logger.info(f"Processing time: {processing_time:.2f} seconds")
    logger.info(f"Performance: {windows_per_second:.1f} windows/second")

    return {
        'total_windows': total_aligned_windows,
        'subject_count': len(subject_counts),
        'batch_count': len(all_windows),
        'processing_time': processing_time,
        'windows_per_second': windows_per_second,
        'modality_counts': {
            'eeg': total_eeg_windows,
            'fnirs': total_fnirs_windows,
            'aligned': total_aligned_windows
        }
    }

def main():
    """Command-line interface for the multimodal window extractor."""
    parser = argparse.ArgumentParser(
        description="Multimodal Window Extractor for STROOP EEG+fNIRS data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--input", type=str, required=True,
                      help="Path to input HDF5 file")
    parser.add_argument("--output", type=str, default="data/processed/extracted_stroop_windows",
                      help="Directory to save extracted windows")
    parser.add_argument("--window-time", type=float, default=2.0,
                      help="Window duration in seconds")
    parser.add_argument("--window-overlap", type=float, default=0.5,
                      help="Overlap fraction between windows (0.0-1.0)")
    parser.add_argument("--windows-per-file", type=int, default=1000,
                      help="Number of windows to save per file")
    parser.add_argument("--num-workers", type=int, default=4,
                      help="Number of worker threads for I/O")
    parser.add_argument("--batch-size", type=int, default=5,
                      help="Number of recordings to process at once")
    parser.add_argument("--cpu", action="store_true",
                      help="Force CPU usage even if GPU is available")

    args = parser.parse_args()

    # Create timestamped output directory
    if not os.path.exists(args.output):
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"{args.output}_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = args.output
        os.makedirs(output_dir, exist_ok=True)

    # Determine device
    device = get_available_device(use_cpu=args.cpu)

    # Extract windows
    stats = extract_stroop_windows_batched(
        hdf5_file=args.input,
        output_dir=output_dir,
        window_time_seconds=args.window_time,
        window_overlap=args.window_overlap,
        windows_per_file=args.windows_per_file,
        num_workers=args.num_workers,
        device=device,
        batch_size=args.batch_size
    )

    # Print summary
    print("\nExtraction Summary:")
    print(f"Total windows extracted: {stats['total_windows']}")
    print(f"Window time: {args.window_time} seconds")
    print(f"Window overlap: {args.window_overlap * 100}%")
    print(f"Subject count: {stats['subject_count']}")
    print(f"Processing time: {stats['processing_time']:.2f} seconds")
    print(f"Performance: {stats['windows_per_second']:.1f} windows/second")
    print(f"Output directory: {output_dir}")

    # Return exit code
    return 0 if stats['total_windows'] > 0 else 1

if __name__ == "__main__":
    exit(main())
