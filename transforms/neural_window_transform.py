"""
Neural Windowing Transform for processing EEG and fNIRS data.

This transform:
1. Downloads data from S3 bucket
2. Applies pre-window preprocessing to EEG data (bandpass, etc.)
3. Applies pre-window preprocessing to fNIRS data
4. Creates time-aligned windows from the preprocessed data
5. Applies post-window processing (normalization, etc.)
6. Stores the windowed data and metadata
"""

import os
import sys
import boto3
import h5py
import numpy as np
from typing import Dict, Any, List, Optional, Set
import json
import time

# Import system modules
import sys
import os

# Determine file locations
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

# Import base transform
from base import DataTransform

# Import neural processing modules
neural_processing_dir = os.path.join(current_dir, 'neural_processing')
sys.path.insert(0, neural_processing_dir)

# Direct imports from the neural_processing directory
from eeg_preprocessing import preprocess_eeg, expand_eeg_timestamps
from fnirs_preprocessing import preprocess_fnirs
from windowing import create_time_aligned_windows
from postprocessing import postprocess_windows
from window_dataset import WindowDataset


class NeuralWindowTransform(DataTransform):
    """
    Neural windowing transform for processing EEG and fNIRS data.

    This transform processes neural data through several stages:
    1. Artefact rejection
    2. Pre-window preprocessing (EEG and fNIRS)
    3. Time-aligned windowing
    4. Post-window processing
    5. Storage and metadata recording
    """

    def __init__(self, source_prefix: str = 'curated-h5/',
                dest_prefix: str = 'processed/windows/',
                window_size_sec: float = 5.0,
                window_step_sec: float = 2.5,
                **kwargs):
        """
        Initialize the neural windowing transform.

        Args:
            source_prefix: S3 prefix for source data
            dest_prefix: S3 prefix for destination data
            window_size_sec: Window size in seconds
            window_step_sec: Window step size in seconds
            **kwargs: Additional arguments for DataTransform
        """
        # Set default transform info if not provided
        transform_id = kwargs.pop('transform_id', 'neural_window_v0')
        script_id = kwargs.pop('script_id', '0B')
        script_name = kwargs.pop('script_name', 'neural_window')
        script_version = kwargs.pop('script_version', 'v0')

        # Call parent constructor
        super().__init__(
            transform_id=transform_id,
            script_id=script_id,
            script_name=script_name,
            script_version=script_version,
            **kwargs
        )

        # Set windowing-specific attributes
        self.source_prefix = source_prefix
        self.dest_prefix = dest_prefix
        self.window_size_sec = window_size_sec
        self.window_step_sec = window_step_sec

        self.logger.info(f"Neural windowing transform initialized with:")
        self.logger.info(f"  Source prefix: {self.source_prefix}")
        self.logger.info(f"  Destination prefix: {self.dest_prefix}")
        self.logger.info(f"  Window size: {self.window_size_sec} seconds")
        self.logger.info(f"  Window step: {self.window_step_sec} seconds")

    def find_items_to_process(self):
        """
        Find H5 files that need to be windowed.

        Returns:
            List of H5 file keys to process
        """
        self.logger.info(f"Listing H5 files in {self.source_prefix}")

        h5_files = []
        paginator = self.s3.get_paginator('list_objects_v2')

        for page in paginator.paginate(Bucket=self.s3_bucket, Prefix=self.source_prefix):
            if 'Contents' in page:
                for obj in page['Contents']:
                    key = obj['Key']
                    if key.endswith('.h5'):
                        h5_files.append(key)

        self.logger.info(f"Found {len(h5_files)} H5 files")
        return h5_files

    def download_h5_file(self, h5_key):
        """
        Download an H5 file from S3 to a temporary local file.

        Args:
            h5_key: S3 key for the H5 file

        Returns:
            Path to the local file
        """
        # Create a temporary local file
        local_path = os.path.join('/tmp', os.path.basename(h5_key))

        try:
            self.logger.info(f"Downloading {h5_key} to {local_path}")

            # Download the file (even in dry run mode to inspect structure)
            start_time = time.time()
            self.s3.download_file(self.s3_bucket, h5_key, local_path)
            download_time = time.time() - start_time

            file_size_mb = os.path.getsize(local_path) / (1024 * 1024)
            self.logger.info(f"Downloaded {file_size_mb:.2f} MB in {download_time:.2f} seconds")

            if self.dry_run:
                self.logger.info(f"[DRY RUN] Downloaded {h5_key} for inspection only")

            return local_path

        except Exception as e:
            self.logger.error(f"Error downloading file {h5_key}: {e}")
            if self.dry_run:
                self.logger.warning(f"[DRY RUN] Would fail with download error: {e}")
                # In dry run, create an empty file for error handling
                with open(local_path, 'wb') as f:
                    f.write(b'')
                return local_path
            else:
                raise

    def upload_windowed_data(self, h5_key, window_dataset):
        """
        Upload windowed data to S3.

        Args:
            h5_key: Original H5 file key
            window_dataset: WindowDataset object

        Returns:
            Dict with destination paths and metadata
        """
        # Generate destination key from source key
        base_name = os.path.basename(h5_key).replace('.h5', '')
        dest_key = f"{self.dest_prefix}{base_name}_windowed.h5"
        dest_path = f"s3://{self.s3_bucket}/{dest_key}"

        # Create a temporary local file for the windowed data
        local_path = os.path.join('/tmp', os.path.basename(dest_key))

        if self.dry_run:
            self.logger.info(f"[DRY RUN] Would create windowed H5 file with {len(window_dataset)} windows")

            # Log details about the window data structure at debug level
            max_windows_to_log = min(3, len(window_dataset))
            if max_windows_to_log > 0:
                self.logger.debug(f"  - Window data structure:")
                for i in range(max_windows_to_log):
                    window_data = window_dataset[i]
                    eeg_shape = window_data['eeg'].shape if hasattr(window_data['eeg'], 'shape') else "unknown"
                    fnirs_shape = window_data['fnirs'].shape if hasattr(window_data['fnirs'], 'shape') else "unknown"
                    self.logger.debug(f"    Window {i}: EEG shape={eeg_shape}, fNIRS shape={fnirs_shape}")

            self.logger.debug(f"[DRY RUN] Would upload data to: {dest_path}")

            # In dry run mode, create a real file with actual data to validate the full pipeline
            try:
                # Save the windowed data to validate the format
                start_time = time.time()
                with h5py.File(local_path, 'w') as f:
                    # Create file-level attributes
                    f.attrs['source_file'] = h5_key
                    f.attrs['window_size_sec'] = self.window_size_sec
                    f.attrs['window_step_sec'] = self.window_step_sec
                    f.attrs['num_windows'] = len(window_dataset)
                    f.attrs['created_at'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
                    f.attrs['dry_run'] = True

                    # Create data groups
                    eeg_group = f.create_group('eeg_windows')
                    fnirs_group = f.create_group('fnirs_windows')
                    metadata_group = f.create_group('metadata')

                    # Store dataset metadata
                    metadata_group.attrs['window_size_sec'] = self.window_size_sec
                    metadata_group.attrs['window_step_sec'] = self.window_step_sec

                    # Store each window (only store the first few in dry run to save time/space)
                    num_windows_to_save = min(5, len(window_dataset))
                    self.logger.debug(f"[DRY RUN] Storing {num_windows_to_save} sample windows to verify format")

                    for i in range(num_windows_to_save):
                        window_data = window_dataset[i]

                        # Store EEG data
                        eeg_group.create_dataset(f"window_{i}", data=window_data['eeg'],
                                               compression="gzip", compression_opts=4)

                        # Store fNIRS data
                        fnirs_group.create_dataset(f"window_{i}", data=window_data['fnirs'],
                                                 compression="gzip", compression_opts=4)

                        # Convert any numpy arrays in metadata to native Python types
                        metadata_copy = {}
                        for k, v in window_data['metadata'].items():
                            if isinstance(v, np.ndarray):
                                metadata_copy[k] = v.tolist()  # Convert numpy array to list
                            elif isinstance(v, np.number):
                                metadata_copy[k] = v.item()    # Convert numpy scalar to native Python type
                            else:
                                metadata_copy[k] = v

                        # Store metadata as JSON string
                        metadata_json = json.dumps(metadata_copy)
                        metadata_group.create_dataset(f"window_{i}", data=metadata_json)

                # Log file size and creation time at debug level
                file_size_mb = os.path.getsize(local_path) / (1024 * 1024)
                creation_time = time.time() - start_time
                self.logger.debug(f"[DRY RUN] Created sample windowed H5 file: {file_size_mb:.2f} MB in {creation_time:.2f} seconds")

                # Log validation success at info level
                self.logger.info(f"[DRY RUN] Successfully validated H5 file format")

                # Clean up in dry run
                os.remove(local_path)
                self.logger.debug(f"[DRY RUN] Cleaned up temporary file")

            except Exception as e:
                self.logger.warning(f"[DRY RUN] Error validating H5 file format: {e}")
                self.logger.debug("Error details:", exc_info=True)

            return {
                "destination_path": dest_path,
                "num_windows": len(window_dataset),
                "status": "dry_run_validated"
            }

        # Save the windowed data to the local file
        self.logger.info(f"Creating windowed H5 file with {len(window_dataset)} windows")

        start_time = time.time()
        with h5py.File(local_path, 'w') as f:
            # Create file-level attributes
            f.attrs['source_file'] = h5_key
            f.attrs['window_size_sec'] = self.window_size_sec
            f.attrs['window_step_sec'] = self.window_step_sec
            f.attrs['num_windows'] = len(window_dataset)
            f.attrs['created_at'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())

            # Create data groups
            eeg_group = f.create_group('eeg_windows')
            fnirs_group = f.create_group('fnirs_windows')
            metadata_group = f.create_group('metadata')

            # Store dataset metadata
            metadata_group.attrs['window_size_sec'] = self.window_size_sec
            metadata_group.attrs['window_step_sec'] = self.window_step_sec

            # Store each window
            for i in range(len(window_dataset)):
                window_data = window_dataset[i]

                # Store EEG data
                eeg_group.create_dataset(f"window_{i}", data=window_data['eeg'],
                                       compression="gzip", compression_opts=4)

                # Store fNIRS data
                fnirs_group.create_dataset(f"window_{i}", data=window_data['fnirs'],
                                         compression="gzip", compression_opts=4)

                # Convert any numpy arrays in metadata to native Python types
                metadata_copy = {}
                for k, v in window_data['metadata'].items():
                    if isinstance(v, np.ndarray):
                        metadata_copy[k] = v.tolist()  # Convert numpy array to list
                    elif isinstance(v, np.number):
                        metadata_copy[k] = v.item()    # Convert numpy scalar to native Python type
                    else:
                        metadata_copy[k] = v

                # Store metadata as JSON string
                metadata_json = json.dumps(metadata_copy)
                metadata_group.create_dataset(f"window_{i}", data=metadata_json)

        # Log file size and creation time
        file_size_mb = os.path.getsize(local_path) / (1024 * 1024)
        creation_time = time.time() - start_time
        self.logger.info(f"Created windowed H5 file: {file_size_mb:.2f} MB in {creation_time:.2f} seconds")

        # Upload the file to S3
        self.logger.info(f"Uploading windowed data to {dest_path}")
        upload_start = time.time()
        self.s3.upload_file(local_path, self.s3_bucket, dest_key)
        upload_time = time.time() - upload_start
        self.logger.info(f"Upload completed in {upload_time:.2f} seconds")

        # Cleanup
        os.remove(local_path)
        self.logger.info("Temporary file removed")

        return {
            "destination_path": dest_path,
            "num_windows": len(window_dataset),
            "file_size_mb": file_size_mb
        }

    def process_h5_file(self, h5_key):
        """
        Process an H5 file to create windowed data.

        Args:
            h5_key: S3 key for the H5 file

        Returns:
            WindowDataset object
        """
        # Download the H5 file (even in dry run mode)
        local_path = self.download_h5_file(h5_key)

        # Try to load the actual file structure and contents
        try:
            self.logger.info(f"Opening H5 file: {h5_key}")

            # Process the H5 file
            with h5py.File(local_path, 'r') as f:
                # Log file structure only at debug level
                self.logger.debug(f"H5 file structure for: {h5_key}")

                def print_structure(name, obj):
                    if hasattr(obj, 'shape'):
                        self.logger.debug(f"  {name}: shape={obj.shape}, dtype={obj.dtype}")
                    else:
                        self.logger.debug(f"  {name}: {type(obj)}")

                f.visititems(print_structure)

                # Extract EEG and fNIRS data
                # Adjust the paths based on actual file structure
                try:
                    # Initialize path lists
                    eeg_data_path = []
                    eeg_time_path = []
                    fnirs_data_path = []
                    fnirs_time_path = []

                    # Standard paths to check
                    standard_paths = [
                        # Standard expected structure
                        ('eeg/data', eeg_data_path),
                        ('eeg/timestamps', eeg_time_path),
                        ('fnirs/data', fnirs_data_path),
                        ('fnirs/timestamps', fnirs_time_path),
                        # Actual structure found in files
                        ('devices/eeg/frames_data', eeg_data_path),
                        ('devices/eeg/timestamps', eeg_time_path),
                        ('devices/fnirs/frames_data', fnirs_data_path),
                        ('devices/fnirs/timestamps', fnirs_time_path)
                    ]

                    # Check if standard paths exist
                    for path, target_list in standard_paths:
                        if path in f:
                            target_list.append(path)
                            self.logger.debug(f"Found standard path: {path}")

                    # If no standard paths were found, try to detect paths automatically
                    if not (eeg_data_path and eeg_time_path and fnirs_data_path and fnirs_time_path):
                        self.logger.debug("Standard paths not found, attempting auto-detection")

                        # Check if 'devices' group exists
                        if 'devices' in f:
                            devices = f['devices']

                            # Look for EEG device data
                            if 'eeg' in devices:
                                eeg_device = devices['eeg']
                                for name in eeg_device:
                                    path = f"devices/eeg/{name}"
                                    if 'data' in name.lower() or 'frame' in name.lower():
                                        eeg_data_path.append(path)
                                        self.logger.debug(f"Auto-detected EEG data path: {path}")
                                    elif 'time' in name.lower():
                                        eeg_time_path.append(path)
                                        self.logger.debug(f"Auto-detected EEG timestamp path: {path}")

                            # Look for fNIRS device data
                            if 'fnirs' in devices:
                                fnirs_device = devices['fnirs']
                                for name in fnirs_device:
                                    path = f"devices/fnirs/{name}"
                                    if 'data' in name.lower() or 'frame' in name.lower():
                                        fnirs_data_path.append(path)
                                        self.logger.debug(f"Auto-detected fNIRS data path: {path}")
                                    elif 'time' in name.lower():
                                        fnirs_time_path.append(path)
                                        self.logger.debug(f"Auto-detected fNIRS timestamp path: {path}")

                        # Also check at root level
                        for name in f:
                            if 'eeg' in name.lower():
                                eeg_group = f[name]
                                if isinstance(eeg_group, h5py.Group):
                                    for subname in eeg_group:
                                        path = f"{name}/{subname}"
                                        if 'data' in subname.lower():
                                            eeg_data_path.append(path)
                                            self.logger.debug(f"Auto-detected EEG data path: {path}")
                                        elif 'time' in subname.lower():
                                            eeg_time_path.append(path)
                                            self.logger.debug(f"Auto-detected EEG timestamp path: {path}")
                            elif 'nir' in name.lower():
                                fnirs_group = f[name]
                                if isinstance(fnirs_group, h5py.Group):
                                    for subname in fnirs_group:
                                        path = f"{name}/{subname}"
                                        if 'data' in subname.lower():
                                            fnirs_data_path.append(path)
                                            self.logger.debug(f"Auto-detected fNIRS data path: {path}")
                                        elif 'time' in subname.lower():
                                            fnirs_time_path.append(path)
                                            self.logger.debug(f"Auto-detected fNIRS timestamp path: {path}")

                    # Ensure we have at least some paths to try
                    if not eeg_data_path:
                        eeg_data_path = ['eeg/data', 'devices/eeg/frames_data', 'devices/eeg/data']
                    if not eeg_time_path:
                        eeg_time_path = ['eeg/timestamps', 'devices/eeg/timestamps']
                    if not fnirs_data_path:
                        fnirs_data_path = ['fnirs/data', 'devices/fnirs/frames_data', 'devices/fnirs/data']
                    if not fnirs_time_path:
                        fnirs_time_path = ['fnirs/timestamps', 'devices/fnirs/timestamps']

                    # Log what we found at debug level
                    self.logger.debug(f"Attempting to load EEG data from: {eeg_data_path}")
                    self.logger.debug(f"Attempting to load EEG timestamps from: {eeg_time_path}")
                    self.logger.debug(f"Attempting to load fNIRS data from: {fnirs_data_path}")
                    self.logger.debug(f"Attempting to load fNIRS timestamps from: {fnirs_time_path}")

                    # Load data (using first path that works)
                    # Load EEG data
                    eeg_data = None
                    eeg_timestamps = None
                    for path in eeg_data_path:
                        if path in f:
                            eeg_data = f[path][:]
                            self.logger.debug(f"Successfully loaded EEG data from {path} with shape {eeg_data.shape}")
                            break

                    for path in eeg_time_path:
                        if path in f:
                            eeg_timestamps = f[path][:]
                            self.logger.debug(f"Successfully loaded EEG timestamps from {path} with shape {eeg_timestamps.shape}")
                            break

                    # Load fNIRS data
                    fnirs_data = None
                    fnirs_timestamps = None
                    for path in fnirs_data_path:
                        if path in f:
                            fnirs_data = f[path][:]
                            self.logger.debug(f"Successfully loaded fNIRS data from {path} with shape {fnirs_data.shape}")
                            break

                    for path in fnirs_time_path:
                        if path in f:
                            fnirs_timestamps = f[path][:]
                            self.logger.debug(f"Successfully loaded fNIRS timestamps from {path} with shape {fnirs_timestamps.shape}")
                            break

                    # Check what data is available and whether we can proceed
                    has_eeg = eeg_data is not None and eeg_timestamps is not None
                    has_fnirs = fnirs_data is not None and fnirs_timestamps is not None

                    # At least one modality must be available
                    if not has_eeg and not has_fnirs:
                        missing_data = []
                        if eeg_data is None:
                            missing_data.append("EEG data")
                        if eeg_timestamps is None:
                            missing_data.append("EEG timestamps")
                        if fnirs_data is None:
                            missing_data.append("fNIRS data")
                        if fnirs_timestamps is None:
                            missing_data.append("fNIRS timestamps")

                        if self.dry_run:
                            self.logger.warning(f"[DRY RUN] Missing all required data: {', '.join(missing_data)}")
                            self.logger.warning(f"[DRY RUN] Cannot continue processing without at least one complete modality")
                            self.logger.warning(f"[DRY RUN] Aborting processing for {h5_key}")
                            raise ValueError(f"Missing all required data in the H5 file: {', '.join(missing_data)}")
                        else:
                            self.logger.error(f"Missing all required data: {', '.join(missing_data)}")
                            raise ValueError(f"No complete modality available in the H5 file: {', '.join(missing_data)}")

                    # Log what modalities are available
                    if has_eeg and has_fnirs:
                        self.logger.info("Both EEG and fNIRS data are available for processing")
                    elif has_eeg:
                        self.logger.info("Only EEG data is available for processing, fNIRS will be skipped")
                    elif has_fnirs:
                        self.logger.info("Only fNIRS data is available for processing, EEG will be skipped")

                    # Extract metadata
                    session_id = h5_key.split('/')[-1].split('.')[0]

                    metadata = {
                        'session_id': session_id,
                        'window_size_sec': self.window_size_sec,
                        'window_step_sec': self.window_step_sec,
                        'has_eeg': has_eeg,
                        'has_fnirs': has_fnirs,
                    }

                    # Add modality-specific metadata only for available modalities
                    if has_eeg:
                        metadata.update({
                            'eeg_sample_rate': len(eeg_timestamps) / (eeg_timestamps[-1] - eeg_timestamps[0]) if len(eeg_timestamps) > 1 else 0,
                            'eeg_channels': eeg_data.shape[1] if len(eeg_data.shape) > 1 else 1,
                        })

                    if has_fnirs:
                        metadata.update({
                            'fnirs_sample_rate': len(fnirs_timestamps) / (fnirs_timestamps[-1] - fnirs_timestamps[0]) if len(fnirs_timestamps) > 1 else 0,
                            'fnirs_channels': fnirs_data.shape[1] if len(fnirs_data.shape) > 1 else 1,
                        })

                    self.logger.info(f"Extracted metadata: {metadata}")

                    # For dry run, log what would happen next, but continue with processing
                    # to capture any warnings and allow for validation
                    if self.dry_run:
                        self.logger.info("[DRY RUN] Data validation successful")
                        self.logger.info("[DRY RUN] Will proceed with the following steps to validate the pipeline:")
                        if has_eeg:
                            self.logger.info(f"  1. Preprocessing EEG data ({eeg_data.shape})")
                        else:
                            self.logger.info(f"  1. Skipping EEG preprocessing (no data available)")

                        if has_fnirs:
                            self.logger.info(f"  2. Preprocessing fNIRS data ({fnirs_data.shape})")
                        else:
                            self.logger.info(f"  2. Skipping fNIRS preprocessing (no data available)")

                        self.logger.info(f"  3. Creating time-aligned windows with size={self.window_size_sec}s, step={self.window_step_sec}s")
                        self.logger.info(f"  4. Applying post-window processing")
                        self.logger.info(f"  5. Creating a WindowDataset")
                        self.logger.info(f"[DRY RUN] No data will be uploaded to S3, but processing will continue for validation")

                    # Process data through the pipeline
                    # Initialize variables
                    processed_eeg = None
                    expanded_eeg_timestamps = None
                    processed_fnirs = None

                    # Log modality presence at INFO level (always visible)
                    modality_str = "Modalities present: " + (
                        f"EEG {'✓' if has_eeg else '✗'}, fNIRS {'✓' if has_fnirs else '✗'}"
                    )
                    self.logger.info(modality_str)

                    # Step 2: EEG preprocessing (if available)
                    if has_eeg:
                        self.logger.debug("Preprocessing EEG data") # Set to debug level
                        processed_eeg, eeg_preprocessing_metadata = preprocess_eeg(eeg_data, metadata)
                        metadata.update(eeg_preprocessing_metadata)
                    else:
                        self.logger.debug("Skipping EEG preprocessing - no EEG data available") # Set to debug level
                        metadata['eeg_preprocessing_skipped'] = True

                        # Create placeholder data for windowing
                        processed_eeg = np.zeros((1, 1), dtype=np.float32)  # Minimal array

                        # Add basic metadata for skipped EEG
                        metadata.update({
                            'eeg_preprocessing_skipped': True,
                            'eeg_channels': 0
                        })

                    # Step 2b: Expand EEG timestamps to match processed data (if available)
                    if has_eeg:
                        self.logger.debug("Expanding EEG timestamps to match processed data") # Set to debug level
                        expanded_eeg_timestamps, timestamp_metadata = expand_eeg_timestamps(eeg_timestamps, metadata)
                        metadata.update(timestamp_metadata)
                        self.logger.debug(f"Expanded timestamps from {eeg_timestamps.shape} to {expanded_eeg_timestamps.shape}") # Set to debug level
                    else:
                        # Create placeholder timestamps for windowing
                        expanded_eeg_timestamps = np.array([0], dtype=np.float64)  # Minimal timestamp

                    # Step 3: fNIRS preprocessing with hardcoded channel filtering (if available)
                    if has_fnirs:
                        self.logger.info("Preprocessing fNIRS data with spatial filtering") # Keep at INFO level

                        # Load layout data if available
                        layout_json_path = os.path.join(neural_processing_dir, 'layout.json')
                        layout_data = None
                        if os.path.exists(layout_json_path):
                            try:
                                with open(layout_json_path, 'r') as f:
                                    layout_data = json.load(f)
                                    self.logger.debug("Loaded fNIRS layout data for channel filtering") # Set to debug level
                            except Exception as e:
                                self.logger.warning(f"Could not load layout data: {e}")

                        # Hardcoded modules list as requested
                        hardcoded_modules = [1, 2, 3, 4, 5, 6, 7, 8, 13, 14, 19, 20, 23, 24, 25, 26, 29, 30, 31, 32]

                        self.logger.info(f"Using hardcoded modules: {hardcoded_modules}") # Keep at INFO level

                        # Pass modules to the spatial filtering algorithm
                        processed_fnirs, fnirs_preprocessing_metadata = preprocess_fnirs(
                            fnirs_data,
                            metadata,
                            layout_data=layout_data,
                            present_modules=hardcoded_modules,
                            max_distance_mm=58  # Use 60mm as the standard distance threshold
                        )

                        # Get the feasible channel indices generated by the algorithm
                        feasible_indices = fnirs_preprocessing_metadata.get('feasible_channel_indices', [])

                        # Log filtering results
                        if feasible_indices:
                            # Log detailed channel info at debug level
                            self.logger.debug(f"Feasible channel indices (first 10): {feasible_indices[:10]}...")

                            # Update metadata correctly - no need to override algorithm's results
                            fnirs_preprocessing_metadata.update({
                                'spatial_filtering_applied': True,
                                'used_modules': hardcoded_modules,
                                'module_count': len(hardcoded_modules)
                            })

                            # Use the data that the algorithm already processed
                            # No need to manually extract channels as preprocessing function already did that
                            if processed_fnirs is not None and len(processed_fnirs.shape) == 3:
                                # Just update the metadata with actual processed shape
                                fnirs_preprocessing_metadata['processed_channels'] = processed_fnirs.shape[1]
                            else:
                                self.logger.warning("Unexpected shape for processed fNIRS data")

                        metadata.update(fnirs_preprocessing_metadata)

                        # Log channel selection results at INFO level (always visible)
                        used_channels = fnirs_preprocessing_metadata.get('used_channel_count', 0)
                        excluded_valid = fnirs_preprocessing_metadata.get('channels_with_valid_data_excluded', 0)
                        all_invalid = fnirs_preprocessing_metadata.get('all_excluded_channels_invalid', False)

                        # Add more detailed information about excluded channels
                        if excluded_valid > 0:
                            self.logger.info(f"fNIRS channels: Using {used_channels} channels from selected modules ({excluded_valid} channels with valid data were excluded)")
                        elif all_invalid:
                            self.logger.info(f"fNIRS channels: Using {used_channels} channels from selected modules (all excluded channels contained only invalid data)")
                        else:
                            self.logger.info(f"fNIRS channels: Using {used_channels} channels from selected modules")

                        # Log distance statistics at debug level
                        try:
                            distance_stats = fnirs_preprocessing_metadata.get('distance_stats', {})
                            self.logger.debug(f"fNIRS distance stats: {distance_stats}")
                        except Exception as e:
                            self.logger.debug(f"Could not log distance stats: {e}")
                    else:
                        # No fNIRS data, create placeholder
                        self.logger.debug("Skipping fNIRS preprocessing - no fNIRS data available") # Set to debug level

                        # Create placeholder data for windowing
                        processed_fnirs = np.zeros((1, 1, 1), dtype=np.float32)  # Create minimal 3D array
                        if fnirs_timestamps is None:
                            fnirs_timestamps = np.array([0], dtype=np.float64)  # Create minimal timestamp

                        # Add metadata for skipped fNIRS
                        fnirs_preprocessing_metadata = {
                            'fnirs_preprocessing_skipped': True,
                            'fnirs_channels': 0,
                            'spatial_filtering_applied': False
                        }
                        metadata.update(fnirs_preprocessing_metadata)

                    # Step 4: Create time-aligned windows (with available modalities)
                    self.logger.debug("Creating time-aligned windows") # Set to debug level

                    # Add flags for which modalities are actually used in windowing
                    metadata['windows_include_eeg'] = has_eeg
                    metadata['windows_include_fnirs'] = has_fnirs

                    # Create windows with whatever data we have (real or placeholders)
                    windowed_eeg, windowed_fnirs, window_metadata = create_time_aligned_windows(
                        processed_eeg, processed_fnirs, expanded_eeg_timestamps, fnirs_timestamps, metadata
                    )
                    self.logger.info(f"Created {len(windowed_eeg)} windows")

                    # Add modality information to window metadata
                    for window_meta in window_metadata:
                        window_meta['has_eeg'] = has_eeg
                        window_meta['has_fnirs'] = has_fnirs

                    # Step 5: Apply post-window processing
                    self.logger.debug("Applying post-window processing") # Set to debug level
                    final_eeg_windows, final_fnirs_windows, final_metadata = postprocess_windows(
                        windowed_eeg, windowed_fnirs, window_metadata
                    )

                    # Create and return the WindowDataset
                    window_dataset = WindowDataset(
                        eeg_windows=final_eeg_windows,
                        fnirs_windows=final_fnirs_windows,
                        metadata=final_metadata
                    )

                    self.logger.info(f"Created WindowDataset with {len(window_dataset)} windows")

                    return window_dataset

                except Exception as e:
                    self.logger.error(f"Error extracting data from H5 file: {e}")
                    if self.dry_run:
                        self.logger.warning(f"[DRY RUN] Would fail with error: {e}")
                        self.logger.warning("[DRY RUN] Cannot proceed with processing in dry run mode due to data extraction issues")
                        self.logger.warning(f"[DRY RUN] H5 file appears to be in an incompatible format")
                    raise

        except Exception as e:
            self.logger.error(f"Error processing H5 file: {e}")
            # Clean up
            if os.path.exists(local_path):
                os.remove(local_path)

            if self.dry_run:
                self.logger.warning(f"[DRY RUN] Would fail with error: {e}")
                self.logger.warning("[DRY RUN] Processing could not be completed due to errors")
                self.logger.warning(f"[DRY RUN] This file would be skipped in a real run")

            raise

        finally:
            # Clean up
            if os.path.exists(local_path):
                os.remove(local_path)

    def process_item(self, h5_key):
        """
        Process a single H5 file.

        Args:
            h5_key: S3 key for the H5 file

        Returns:
            Dict with processing result
        """
        # Extract session_id from the H5 key
        session_id = h5_key.split('/')[-1].split('.')[0]

        try:
            self.logger.info(f"Processing file: {h5_key}")
            self.logger.info(f"Processing session ID: {session_id}")

            # Process the H5 file to create a WindowDataset
            start_time = time.time()

            try:
                window_dataset = self.process_h5_file(h5_key)
                processing_time = time.time() - start_time
                self.logger.info(f"Processing completed in {processing_time:.2f} seconds")
            except Exception as e:
                # Real error during processing
                self.logger.error(f"Error processing H5 file: {e}")
                self.logger.debug("Error details:", exc_info=True)
                raise

            # Process differently based on run mode
            if self.dry_run:
                # In dry run mode, we've already processed the data for validation
                processing_time = time.time() - start_time
                self.logger.info(f"[DRY RUN] Validation completed in {processing_time:.2f} seconds")

                # Log what would be written to DynamoDB in dry run mode (at debug level)
                self.logger.debug(f"[DRY RUN] Would record the following metadata to DynamoDB:")
                self.logger.debug(f"  - data_id: {session_id}")
                self.logger.debug(f"  - transform_id: {self.transform_id}")
                self.logger.debug(f"  - window_size_sec: {self.window_size_sec}")
                self.logger.debug(f"  - window_step_sec: {self.window_step_sec}")
                self.logger.debug(f"  - processed_at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
                # Add a simple info level message
                self.logger.info(f"[DRY RUN] Validated processing for session {session_id}")

                # Upload will be skipped in dry run mode, but we call it for validation
                upload_result = self.upload_windowed_data(h5_key, window_dataset)
            else:
                # For normal operation, upload the windowed data to S3
                upload_result = self.upload_windowed_data(h5_key, window_dataset)

            # Record window metadata in DynamoDB
            windows_metadata = []
            for i in range(len(window_dataset)):
                window_data = window_dataset[i]
                window_metadata = window_data['metadata'].copy()

                # Convert NumPy arrays and scalars to native Python types for DynamoDB
                serialized_metadata = {}
                for k, v in window_metadata.items():
                    if isinstance(v, np.ndarray):
                        serialized_metadata[k] = v.tolist()  # Convert numpy array to list
                    elif isinstance(v, np.number):
                        serialized_metadata[k] = v.item()    # Convert numpy scalar to native Python type
                    else:
                        serialized_metadata[k] = v

                serialized_metadata['window_idx'] = i
                windows_metadata.append(serialized_metadata)

            # Prepare metadata for DynamoDB
            transform_metadata = {
                'window_size_sec': self.window_size_sec,
                'window_step_sec': self.window_step_sec,
                'num_windows': len(window_dataset),
                'windows': windows_metadata,
                'processing_time_sec': processing_time,
                'processed_at': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
            }

            # Add modality info if available in the metadata
            # This will be present if we went through our modified process_h5_file
            sample_window = window_dataset[0] if len(window_dataset) > 0 else None
            if sample_window and 'metadata' in sample_window:
                window_meta = sample_window['metadata']
                if 'has_eeg' in window_meta:
                    transform_metadata['has_eeg'] = window_meta['has_eeg']
                if 'has_fnirs' in window_meta:
                    transform_metadata['has_fnirs'] = window_meta['has_fnirs']

            # Record the transform
            transform_result = self.record_transform(
                data_id=session_id,
                transform_metadata=transform_metadata,
                source_paths=[f"s3://{self.s3_bucket}/{h5_key}"],
                destination_paths=[upload_result["destination_path"]],
                status='success'
            )

            return transform_result or {"status": "skipped"}

        except Exception as e:
            self.logger.error(f"Error processing file {h5_key}: {e}")
            self.logger.debug("Error details:", exc_info=True)

            # Log what would be recorded to DynamoDB in case of error
            if self.dry_run:
                self.logger.info(f"[DRY RUN] Would record the following error metadata to DynamoDB:")
                self.logger.info(f"  - data_id: {session_id}")
                self.logger.info(f"  - transform_id: {self.transform_id}")
                self.logger.info(f"  - status: failed")
                self.logger.info(f"  - error_details: {str(e)}")

                # In dry run mode, return information about the error but don't record it
                return {
                    "status": "dry_run_failed",
                    "data_id": session_id,
                    "error": str(e)
                }

            # Record the failure
            transform_result = self.record_transform(
                data_id=session_id,
                transform_metadata={
                    'window_size_sec': self.window_size_sec,
                    'window_step_sec': self.window_step_sec,
                    'processed_at': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
                },
                source_paths=[f"s3://{self.s3_bucket}/{h5_key}"],
                status='failed',
                error_details=str(e)
            )

            return {
                "status": "failed",
                "error": str(e)
            }

    @classmethod
    def add_subclass_arguments(cls, parser):
        """
        Add neural windowing-specific command-line arguments.

        Args:
            parser: ArgumentParser instance
        """
        parser.add_argument('--source-prefix', type=str, default='curated-h5/',
                          help='S3 prefix for source data')
        parser.add_argument('--dest-prefix', type=str, default='processed/windows/',
                          help='S3 prefix for destination data')
        parser.add_argument('--window-size', type=float, default=5.0,
                          help='Window size in seconds')
        parser.add_argument('--window-step', type=float, default=2.5,
                          help='Window step size in seconds')
        parser.add_argument('--s3-bucket', type=str, default='conduit-data-dev',
                          help='S3 bucket name')

    @classmethod
    def from_args(cls, args):
        """
        Create a transform instance from command-line arguments.

        Args:
            args: Parsed command-line arguments

        Returns:
            Instance of NeuralWindowTransform
        """
        return cls(
            source_prefix=args.source_prefix,
            dest_prefix=args.dest_prefix,
            window_size_sec=args.window_size,
            window_step_sec=args.window_step,
            s3_bucket=args.s3_bucket,
            verbose=args.verbose,
            log_file=args.log_file,
            dry_run=args.dry_run
        )


# Entry point for running the transform from the command line
if __name__ == "__main__":
    NeuralWindowTransform.run_from_command_line()
