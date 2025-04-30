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
from eeg_preprocessing import preprocess_eeg
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
                dest_prefix: str = 'windowed-data/',
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
            self.logger.info(f"[DRY RUN] Would create windowed H5 file with:")
            self.logger.info(f"  - Number of windows: {len(window_dataset)}")

            # Log details about the first few windows
            max_windows_to_log = min(3, len(window_dataset))
            if max_windows_to_log > 0:
                self.logger.info(f"  - Window data structure:")
                for i in range(max_windows_to_log):
                    window_data = window_dataset[i]
                    eeg_shape = window_data['eeg'].shape if hasattr(window_data['eeg'], 'shape') else "unknown"
                    fnirs_shape = window_data['fnirs'].shape if hasattr(window_data['fnirs'], 'shape') else "unknown"
                    self.logger.info(f"    Window {i}: EEG shape={eeg_shape}, fNIRS shape={fnirs_shape}")

            self.logger.info(f"[DRY RUN] Would upload data to: {dest_path}")

            # In dry run mode, we still create the file structure but don't upload it
            try:
                # Create a minimal HDF5 file to test the structure
                with h5py.File(local_path, 'w') as f:
                    # Create basic structure
                    f.create_group('eeg_windows')
                    f.create_group('fnirs_windows')
                    f.create_group('metadata')

                    # Add file-level attributes for easier identification
                    f.attrs['source_file'] = h5_key
                    f.attrs['window_size_sec'] = self.window_size_sec
                    f.attrs['window_step_sec'] = self.window_step_sec
                    f.attrs['num_windows'] = len(window_dataset)
                    f.attrs['created_at'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
                    f.attrs['dry_run'] = True

                file_size_mb = os.path.getsize(local_path) / (1024 * 1024)
                self.logger.info(f"[DRY RUN] Verified H5 file structure ({file_size_mb:.2f} MB)")

                # Clean up in dry run too
                os.remove(local_path)
                self.logger.info(f"[DRY RUN] Cleaned up temporary file")

            except Exception as e:
                self.logger.warning(f"[DRY RUN] Error testing H5 file structure: {e}")

            return {
                "destination_path": dest_path,
                "num_windows": len(window_dataset)
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

                # Store metadata as JSON string
                metadata_json = json.dumps(window_data['metadata'])
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
                # Log file structure for debugging
                self.logger.info(f"H5 file structure for: {h5_key}")

                def print_structure(name, obj):
                    if hasattr(obj, 'shape'):
                        self.logger.info(f"  {name}: shape={obj.shape}, dtype={obj.dtype}")
                    else:
                        self.logger.info(f"  {name}: {type(obj)}")

                f.visititems(print_structure)

                # Extract EEG and fNIRS data
                # Adjust the paths based on actual file structure
                try:
                    # Try to locate EEG data
                    eeg_paths = [k for k in f.keys() if 'eeg' in k.lower()]
                    eeg_data_path = [k for k in eeg_paths if 'data' in k.lower()] if eeg_paths else ['eeg/data']
                    eeg_time_path = [k for k in eeg_paths if 'time' in k.lower()] if eeg_paths else ['eeg/timestamps']

                    # Try to locate fNIRS data
                    fnirs_paths = [k for k in f.keys() if 'nir' in k.lower()]
                    fnirs_data_path = [k for k in fnirs_paths if 'data' in k.lower()] if fnirs_paths else ['fnirs/data']
                    fnirs_time_path = [k for k in fnirs_paths if 'time' in k.lower()] if fnirs_paths else ['fnirs/timestamps']

                    # Log what we found
                    self.logger.info(f"Attempting to load EEG data from: {eeg_data_path}")
                    self.logger.info(f"Attempting to load EEG timestamps from: {eeg_time_path}")
                    self.logger.info(f"Attempting to load fNIRS data from: {fnirs_data_path}")
                    self.logger.info(f"Attempting to load fNIRS timestamps from: {fnirs_time_path}")

                    # Load data (using first path that works)
                    # Load EEG data
                    eeg_data = None
                    eeg_timestamps = None
                    for path in eeg_data_path:
                        if path in f:
                            eeg_data = f[path][:]
                            self.logger.info(f"Successfully loaded EEG data from {path} with shape {eeg_data.shape}")
                            break

                    for path in eeg_time_path:
                        if path in f:
                            eeg_timestamps = f[path][:]
                            self.logger.info(f"Successfully loaded EEG timestamps from {path} with shape {eeg_timestamps.shape}")
                            break

                    # Load fNIRS data
                    fnirs_data = None
                    fnirs_timestamps = None
                    for path in fnirs_data_path:
                        if path in f:
                            fnirs_data = f[path][:]
                            self.logger.info(f"Successfully loaded fNIRS data from {path} with shape {fnirs_data.shape}")
                            break

                    for path in fnirs_time_path:
                        if path in f:
                            fnirs_timestamps = f[path][:]
                            self.logger.info(f"Successfully loaded fNIRS timestamps from {path} with shape {fnirs_timestamps.shape}")
                            break

                    # Check if we have all necessary data
                    missing_data = []
                    if eeg_data is None:
                        missing_data.append("EEG data")
                    if eeg_timestamps is None:
                        missing_data.append("EEG timestamps")
                    if fnirs_data is None:
                        missing_data.append("fNIRS data")
                    if fnirs_timestamps is None:
                        missing_data.append("fNIRS timestamps")

                    if missing_data:
                        if self.dry_run:
                            self.logger.warning(f"[DRY RUN] Missing required data: {', '.join(missing_data)}")
                            self.logger.warning(f"[DRY RUN] Cannot continue processing without this data")
                            self.logger.warning(f"[DRY RUN] Aborting processing for {h5_key}")
                            raise ValueError(f"Missing required data in the H5 file: {', '.join(missing_data)}")
                        else:
                            self.logger.error(f"Missing required data: {', '.join(missing_data)}")
                            raise ValueError(f"Could not find all required data in the H5 file: {', '.join(missing_data)}")

                    # Extract metadata
                    session_id = h5_key.split('/')[-1].split('.')[0]

                    metadata = {
                        'session_id': session_id,
                        'window_size_sec': self.window_size_sec,
                        'window_step_sec': self.window_step_sec,
                        'eeg_sample_rate': len(eeg_timestamps) / (eeg_timestamps[-1] - eeg_timestamps[0]) if len(eeg_timestamps) > 1 else 0,
                        'fnirs_sample_rate': len(fnirs_timestamps) / (fnirs_timestamps[-1] - fnirs_timestamps[0]) if len(fnirs_timestamps) > 1 else 0,
                        'eeg_channels': eeg_data.shape[1] if len(eeg_data.shape) > 1 else 1,
                        'fnirs_channels': fnirs_data.shape[1] if len(fnirs_data.shape) > 1 else 1
                    }

                    self.logger.info(f"Extracted metadata: {metadata}")

                    # For dry run, stop here and log what would happen next
                    if self.dry_run:
                        self.logger.info("[DRY RUN] Data validation successful")
                        self.logger.info("[DRY RUN] Would proceed with the following steps:")
                        self.logger.info(f"  1. Preprocessing EEG data ({eeg_data.shape})")
                        self.logger.info(f"  2. Preprocessing fNIRS data ({fnirs_data.shape})")
                        self.logger.info(f"  3. Creating time-aligned windows with size={self.window_size_sec}s, step={self.window_step_sec}s")
                        self.logger.info(f"  4. Applying post-window processing")
                        self.logger.info(f"  5. Creating a WindowDataset")
                        self.logger.info(f"[DRY RUN] Skipping actual processing in dry run mode")

                        # Return None in dry run mode - the caller must handle this case
                        raise ValueError("Dry run validation completed, stopping processing as requested")

                    # Process data through the pipeline
                    # Step 2: EEG preprocessing
                    self.logger.info("Preprocessing EEG data")
                    processed_eeg, eeg_preprocessing_metadata = preprocess_eeg(eeg_data, metadata)
                    metadata.update(eeg_preprocessing_metadata)

                    # Step 3: fNIRS preprocessing
                    self.logger.info("Preprocessing fNIRS data")
                    processed_fnirs, fnirs_preprocessing_metadata = preprocess_fnirs(fnirs_data, metadata)
                    metadata.update(fnirs_preprocessing_metadata)

                    # Step 4: Create time-aligned windows
                    self.logger.info("Creating time-aligned windows")
                    windowed_eeg, windowed_fnirs, window_metadata = create_time_aligned_windows(
                        processed_eeg, processed_fnirs, eeg_timestamps, fnirs_timestamps, metadata
                    )
                    self.logger.info(f"Created {len(windowed_eeg)} windows")

                    # Step 5: Apply post-window processing
                    self.logger.info("Applying post-window processing")
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
            except ValueError as e:
                if self.dry_run and "Dry run validation completed" in str(e):
                    # This is the expected path in dry run - validation succeeded
                    processing_time = time.time() - start_time
                    self.logger.info(f"[DRY RUN] Validation completed in {processing_time:.2f} seconds")

                    # Log what would be written to DynamoDB in dry run mode
                    self.logger.info(f"[DRY RUN] Would record the following metadata to DynamoDB:")
                    self.logger.info(f"  - data_id: {session_id}")
                    self.logger.info(f"  - transform_id: {self.transform_id}")
                    self.logger.info(f"  - window_size_sec: {self.window_size_sec}")
                    self.logger.info(f"  - window_step_sec: {self.window_step_sec}")
                    self.logger.info(f"  - processed_at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")

                    # Return what the simulation would produce
                    return {
                        "status": "dry_run_validated",
                        "data_id": session_id,
                        "message": "Dry run validation completed successfully"
                    }
                else:
                    # Real error during processing
                    raise

            # If we get here, it means we're in normal run mode with successful processing

            # Upload the windowed data to S3
            upload_result = self.upload_windowed_data(h5_key, window_dataset)

            # Record window metadata in DynamoDB
            windows_metadata = []
            for i in range(len(window_dataset)):
                window_data = window_dataset[i]
                metadata = window_data['metadata'].copy()
                metadata['window_idx'] = i
                windows_metadata.append(metadata)

            # Prepare metadata for DynamoDB
            transform_metadata = {
                'window_size_sec': self.window_size_sec,
                'window_step_sec': self.window_step_sec,
                'num_windows': len(window_dataset),
                'windows': windows_metadata,
                'processing_time_sec': processing_time,
                'processed_at': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
            }

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
        parser.add_argument('--dest-prefix', type=str, default='windowed-data/',
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
