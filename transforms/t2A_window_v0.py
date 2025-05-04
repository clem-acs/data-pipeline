"""
Neural Windowing Transform for processing EEG and fNIRS data.

This transform:
1. Downloads data from S3 bucket (curated sessions)
2. Applies pre-window preprocessing to EEG data (bandpass, etc.)
3. Applies pre-window preprocessing to fNIRS data
4. Creates time-aligned windows from the preprocessed data
5. Applies post-window processing (normalization, etc.)
6. Stores the windowed data and metadata

This is implemented using the new BaseTransform architecture.
"""

import os
import sys
import json
import time
import h5py
import numpy as np
from typing import Dict, Any, List, Optional

# Import base transform
from base_transform import BaseTransform, Session

# Add neural_processing directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
neural_processing_dir = os.path.join(current_dir, 'neural_processing')
sys.path.insert(0, neural_processing_dir)

# Import neural processing modules
from eeg_preprocessing import preprocess_eeg, expand_eeg_timestamps
from fnirs_preprocessing import preprocess_fnirs
from windowing import create_time_aligned_windows
from postprocessing import postprocess_windows
from window_dataset import WindowDataset


class WindowTransform(BaseTransform):
    """
    Neural windowing transform for processing EEG and fNIRS data.

    This transform processes neural data through several stages:
    1. Artefact rejection
    2. Pre-window preprocessing (EEG and fNIRS)
    3. Time-aligned windowing
    4. Post-window processing
    5. Storage and metadata recording
    """

    # Define required class attributes for source and destination
    SOURCE_PREFIX = 'curated-h5/'
    DEST_PREFIX = 'processed/windows/'

    def __init__(self, window_size_sec: float = 5.0,
                 window_step_sec: float = 2.5,
                 **kwargs):
        """
        Initialize the neural windowing transform.

        Args:
            window_size_sec: Window size in seconds
            window_step_sec: Window step size in seconds
            **kwargs: Additional arguments for BaseTransform
        """
        # Set default transform info if not provided
        transform_id = kwargs.pop('transform_id', 'neural_window_v0')
        script_id = kwargs.pop('script_id', '2A')
        script_name = kwargs.pop('script_name', 'window_neural')
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
        self.window_size_sec = window_size_sec
        self.window_step_sec = window_step_sec

        self.logger.info(f"Neural windowing transform initialized with:")
        self.logger.info(f"  Window size: {self.window_size_sec} seconds")
        self.logger.info(f"  Window step: {self.window_step_sec} seconds")

    def process_session(self, session: Session) -> Dict:
        """Process a single session.

        This implementation:
        1. Downloads the H5 file
        2. Extracts and preprocesses EEG and fNIRS data
        3. Creates time-aligned windows
        4. Uploads the windowed data

        Args:
            session: Session object

        Returns:
            Dict with processing results
        """
        session_id = session.session_id
        self.logger.info(f"Processing session: {session_id}")

        # In curated-h5/, files are always directly in the source prefix
        direct_h5_key = f"{self.source_prefix}{session_id}.h5"

        # Check if the file exists
        try:
            self.s3.head_object(Bucket=self.s3_bucket, Key=direct_h5_key)
            self.logger.info(f"Found H5 file: {direct_h5_key}")
            h5_files = [direct_h5_key]
        except Exception as e:
            self.logger.error(f"No H5 file found for session {session_id}: {e}")
            return {
                "status": "failed",
                "error_details": f"No H5 file found for session {session_id}",
                "metadata": {"session_id": session_id},
                "files_to_copy": [],
                "files_to_upload": []
            }

        # For multiple H5 files, use the largest one
        if len(h5_files) > 1:
            self.logger.warning(f"Multiple H5 files found for session {session_id}, using the largest one")
            file_sizes = session.get_file_sizes(extension='.h5')
            h5_file_key = max(h5_files, key=lambda k: file_sizes.get(k, 0))
        else:
            h5_file_key = h5_files[0]

        self.logger.info(f"Using H5 file: {h5_file_key}")

        # Download the H5 file
        local_h5_path = session.download_file(h5_file_key)

        try:
            # Process the H5 file
            self.logger.info(f"Processing H5 file: {h5_file_key}")

            # Process the H5 file and create windows
            window_dataset = self.process_h5_file(local_h5_path, session_id)

            if window_dataset is None or len(window_dataset) == 0:
                self.logger.warning(f"No windows created for session {session_id}")
                return {
                    "status": "skipped",
                    "metadata": {"session_id": session_id, "reason": "No windows created"},
                    "files_to_copy": [],
                    "files_to_upload": []
                }

            # Create a temporary file for the windowed data
            windowed_file_name = f"{session_id}_windowed.h5"
            local_windowed_path = session.create_upload_file(windowed_file_name)

            # Save the windowed data
            self.save_windowed_data(local_windowed_path, window_dataset, h5_file_key)

            # Create metadata
            metadata = self.create_metadata(window_dataset, session_id)

            # Define the destination key
            dest_key = f"{self.destination_prefix}{windowed_file_name}"

            return {
                "status": "success",
                "metadata": metadata,
                "files_to_copy": [],
                "files_to_upload": [(local_windowed_path, dest_key)]
            }

        except Exception as e:
            self.logger.error(f"Error processing session {session_id}: {e}", exc_info=True)
            return {
                "status": "failed",
                "error_details": str(e),
                "metadata": {"session_id": session_id},
                "files_to_copy": [],
                "files_to_upload": []
            }

    def process_h5_file(self, local_path: str, session_id: str) -> Optional[WindowDataset]:
        """Process an H5 file to create windowed data.

        Args:
            local_path: Path to the local H5 file
            session_id: Session ID

        Returns:
            WindowDataset object or None if processing fails
        """
        try:
            with h5py.File(local_path, 'r') as f:
                self.logger.debug(f"H5 file structure:")
                f.visititems(lambda name, obj: self.logger.debug(f"  {name}: {type(obj)}") if hasattr(obj, 'shape') else None)

                # Extract EEG and fNIRS data
                eeg_data, eeg_timestamps, fnirs_data, fnirs_timestamps = self.extract_data_from_h5(f)

                # Check what data is available
                has_eeg = eeg_data is not None and eeg_timestamps is not None
                has_fnirs = fnirs_data is not None and fnirs_timestamps is not None

                # At least one modality must be available
                if not has_eeg and not has_fnirs:
                    self.logger.error("No valid data found in the H5 file")
                    return None

                # Extract metadata
                metadata = {
                    'session_id': session_id,
                    'window_size_sec': self.window_size_sec,
                    'window_step_sec': self.window_step_sec,
                    'has_eeg': has_eeg,
                    'has_fnirs': has_fnirs,
                }

                # Add modality-specific metadata
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

                # Process data through the pipeline
                processed_eeg, expanded_timestamps, processed_fnirs = self.preprocess_data(
                    eeg_data, eeg_timestamps, fnirs_data, fnirs_timestamps, metadata
                )

                # Create time-aligned windows
                return self.create_windows(
                    processed_eeg, processed_fnirs,
                    expanded_timestamps, fnirs_timestamps,
                    metadata
                )

        except Exception as e:
            self.logger.error(f"Error processing H5 file: {e}", exc_info=True)
            return None

    def extract_data_from_h5(self, h5_file):
        """Extract EEG and fNIRS data from the H5 file.

        Args:
            h5_file: h5py File object

        Returns:
            Tuple of (eeg_data, eeg_timestamps, fnirs_data, fnirs_timestamps)
        """
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
            if path in h5_file:
                target_list.append(path)
                self.logger.debug(f"Found standard path: {path}")

        # If no standard paths were found, try to detect paths automatically
        if not (eeg_data_path and eeg_time_path and fnirs_data_path and fnirs_time_path):
            self.logger.debug("Standard paths not found, attempting auto-detection")

            # Check if 'devices' group exists
            if 'devices' in h5_file:
                devices = h5_file['devices']

                # Look for EEG device data
                if 'eeg' in devices:
                    eeg_device = devices['eeg']
                    for name in eeg_device:
                        path = f"devices/eeg/{name}"
                        if 'data' in name.lower() or 'frame' in name.lower():
                            eeg_data_path.append(path)
                        elif 'time' in name.lower():
                            eeg_time_path.append(path)

                # Look for fNIRS device data
                if 'fnirs' in devices:
                    fnirs_device = devices['fnirs']
                    for name in fnirs_device:
                        path = f"devices/fnirs/{name}"
                        if 'data' in name.lower() or 'frame' in name.lower():
                            fnirs_data_path.append(path)
                        elif 'time' in name.lower():
                            fnirs_time_path.append(path)

            # Also check at root level
            for name in h5_file:
                if 'eeg' in name.lower():
                    eeg_group = h5_file[name]
                    if isinstance(eeg_group, h5py.Group):
                        for subname in eeg_group:
                            path = f"{name}/{subname}"
                            if 'data' in subname.lower():
                                eeg_data_path.append(path)
                            elif 'time' in subname.lower():
                                eeg_time_path.append(path)
                elif 'nir' in name.lower():
                    fnirs_group = h5_file[name]
                    if isinstance(fnirs_group, h5py.Group):
                        for subname in fnirs_group:
                            path = f"{name}/{subname}"
                            if 'data' in subname.lower():
                                fnirs_data_path.append(path)
                            elif 'time' in subname.lower():
                                fnirs_time_path.append(path)

        # Ensure we have at least some paths to try
        if not eeg_data_path:
            eeg_data_path = ['eeg/data', 'devices/eeg/frames_data', 'devices/eeg/data']
        if not eeg_time_path:
            eeg_time_path = ['eeg/timestamps', 'devices/eeg/timestamps']
        if not fnirs_data_path:
            fnirs_data_path = ['fnirs/data', 'devices/fnirs/frames_data', 'devices/fnirs/data']
        if not fnirs_time_path:
            fnirs_time_path = ['fnirs/timestamps', 'devices/fnirs/timestamps']

        # Load data (using first path that works)
        # Load EEG data
        eeg_data = None
        eeg_timestamps = None
        for path in eeg_data_path:
            if path in h5_file:
                eeg_data = h5_file[path][:]
                self.logger.debug(f"Loaded EEG data from {path} with shape {eeg_data.shape}")
                break

        for path in eeg_time_path:
            if path in h5_file:
                eeg_timestamps = h5_file[path][:]
                self.logger.debug(f"Loaded EEG timestamps from {path} with shape {eeg_timestamps.shape}")
                break

        # Load fNIRS data
        fnirs_data = None
        fnirs_timestamps = None
        for path in fnirs_data_path:
            if path in h5_file:
                fnirs_data = h5_file[path][:]
                self.logger.debug(f"Loaded fNIRS data from {path} with shape {fnirs_data.shape}")
                break

        for path in fnirs_time_path:
            if path in h5_file:
                fnirs_timestamps = h5_file[path][:]
                self.logger.debug(f"Loaded fNIRS timestamps from {path} with shape {fnirs_timestamps.shape}")
                break

        return eeg_data, eeg_timestamps, fnirs_data, fnirs_timestamps

    def preprocess_data(self, eeg_data, eeg_timestamps, fnirs_data, fnirs_timestamps, metadata):
        """Preprocess EEG and fNIRS data.

        Args:
            eeg_data: EEG data array or None
            eeg_timestamps: EEG timestamps array or None
            fnirs_data: fNIRS data array or None
            fnirs_timestamps: fNIRS timestamps array or None
            metadata: Metadata dictionary

        Returns:
            Tuple of (processed_eeg, expanded_eeg_timestamps, processed_fnirs)
        """
        has_eeg = eeg_data is not None and eeg_timestamps is not None
        has_fnirs = fnirs_data is not None and fnirs_timestamps is not None

        # Process EEG if available
        processed_eeg = None
        expanded_eeg_timestamps = None
        if has_eeg:
            self.logger.info("Preprocessing EEG data")
            processed_eeg, eeg_preprocessing_metadata = preprocess_eeg(eeg_data, metadata)
            metadata.update(eeg_preprocessing_metadata)

            # Expand timestamps
            expanded_eeg_timestamps, timestamp_metadata = expand_eeg_timestamps(eeg_timestamps, metadata)
            metadata.update(timestamp_metadata)
        else:
            # Create placeholders
            processed_eeg = np.zeros((1, 1), dtype=np.float32)
            expanded_eeg_timestamps = np.array([0], dtype=np.float64)
            metadata['eeg_preprocessing_skipped'] = True
            metadata['eeg_channels'] = 0

        # Process fNIRS if available
        processed_fnirs = None
        if has_fnirs:
            self.logger.info("Preprocessing fNIRS data with spatial filtering")

            # Load layout data if available
            layout_json_path = os.path.join(neural_processing_dir, 'layout.json')
            layout_data = None
            if os.path.exists(layout_json_path):
                try:
                    with open(layout_json_path, 'r') as f:
                        layout_data = json.load(f)
                except Exception as e:
                    self.logger.warning(f"Could not load layout data: {e}")

            # Hardcoded modules list
            hardcoded_modules = [1, 2, 3, 4, 5, 6, 7, 8, 13, 14, 19, 20, 23, 24, 25, 26, 29, 30, 31, 32]
            self.logger.info(f"Using hardcoded modules: {hardcoded_modules}")

            # Process fNIRS data
            processed_fnirs, fnirs_preprocessing_metadata = preprocess_fnirs(
                fnirs_data,
                metadata,
                layout_data=layout_data,
                present_modules=hardcoded_modules,
                max_distance_mm=60
            )

            metadata.update(fnirs_preprocessing_metadata)

            # Log enhanced channel statistics
            used_channels = fnirs_preprocessing_metadata.get('used_channel_count', 0)
            included_valid = fnirs_preprocessing_metadata.get('included_channels_with_valid_data', 0)
            included_invalid = fnirs_preprocessing_metadata.get('included_channels_all_invalid', 0)
            excluded_valid = fnirs_preprocessing_metadata.get('excluded_channels_with_valid_data', 0)
            all_invalid = fnirs_preprocessing_metadata.get('all_excluded_channels_invalid', False)
            valid_percentage = fnirs_preprocessing_metadata.get('valid_channel_percentage', 0)

            # Detailed logging of channel quality
            self.logger.info(f"fNIRS channel quality: {included_valid} of {used_channels} included channels ({valid_percentage:.1f}%) contain valid data")

            if included_invalid > 0:
                self.logger.warning(f"fNIRS data quality issue: {included_invalid} included channels contain only invalid data (NaN/Infinity values)")

            # Add information about excluded channels
            if excluded_valid > 0:
                self.logger.info(f"fNIRS spatial filtering: {excluded_valid} channels with valid data were excluded based on distance constraints")
            elif all_invalid:
                self.logger.debug(f"fNIRS spatial filtering: All excluded channels contained only invalid data")

            # Log distance statistics at debug level
            try:
                distance_stats = fnirs_preprocessing_metadata.get('distance_stats', {})
                self.logger.debug(f"fNIRS distance stats: {distance_stats}")
            except Exception as e:
                self.logger.debug(f"Could not log distance stats: {e}")
        else:
            # Create placeholder
            processed_fnirs = np.zeros((1, 1, 1), dtype=np.float32)
            metadata['fnirs_preprocessing_skipped'] = True
            metadata['fnirs_channels'] = 0
            metadata['spatial_filtering_applied'] = False

        return processed_eeg, expanded_eeg_timestamps, processed_fnirs

    def create_windows(self, processed_eeg, processed_fnirs, expanded_eeg_timestamps, fnirs_timestamps, metadata):
        """Create time-aligned windows from the preprocessed data.

        Args:
            processed_eeg: Preprocessed EEG data
            processed_fnirs: Preprocessed fNIRS data
            expanded_eeg_timestamps: Expanded EEG timestamps
            fnirs_timestamps: fNIRS timestamps
            metadata: Metadata dictionary

        Returns:
            WindowDataset object
        """
        # Add flags for which modalities are actually used in windowing
        metadata['windows_include_eeg'] = metadata.get('has_eeg', False)
        metadata['windows_include_fnirs'] = metadata.get('has_fnirs', False)

        # Create windows
        self.logger.info("Creating time-aligned windows")
        windowed_eeg, windowed_fnirs, window_metadata = create_time_aligned_windows(
            processed_eeg, processed_fnirs, expanded_eeg_timestamps, fnirs_timestamps, metadata
        )
        self.logger.info(f"Created {len(windowed_eeg)} windows")

        # Add modality information to window metadata
        for window_meta in window_metadata:
            window_meta['has_eeg'] = metadata.get('has_eeg', False)
            window_meta['has_fnirs'] = metadata.get('has_fnirs', False)

        # Apply post-window processing
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

    def save_windowed_data(self, local_path, window_dataset, source_h5_key):
        """Save the windowed data to a local H5 file.

        Args:
            local_path: Path to save the H5 file
            window_dataset: WindowDataset object
            source_h5_key: Source H5 file key
        """
        self.logger.info(f"Creating windowed H5 file with {len(window_dataset)} windows")

        start_time = time.time()
        with h5py.File(local_path, 'w') as f:
            # Create file-level attributes
            f.attrs['source_file'] = source_h5_key
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

    def create_metadata(self, window_dataset, session_id):
        """Create metadata for the transform.

        Args:
            window_dataset: WindowDataset object
            session_id: Session ID

        Returns:
            Dict with metadata
        """
        # Get sample window metadata
        sample_window = window_dataset[0] if len(window_dataset) > 0 else None
        window_meta = sample_window['metadata'] if sample_window else {}

        # Create metadata for DynamoDB
        return {
            'session_id': session_id,
            'window_size_sec': self.window_size_sec,
            'window_step_sec': self.window_step_sec,
            'num_windows': len(window_dataset),
            'has_eeg': window_meta.get('has_eeg', False),
            'has_fnirs': window_meta.get('has_fnirs', False),
            'processed_at': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        }

    @classmethod
    def add_subclass_arguments(cls, parser):
        """Add neural windowing-specific command-line arguments.

        Args:
            parser: ArgumentParser instance
        """
        parser.add_argument('--window-size', type=float, default=5.0,
                          help='Window size in seconds')
        parser.add_argument('--window-step', type=float, default=2.5,
                          help='Window step size in seconds')

    @classmethod
    def from_args(cls, args):
        """Create a transform instance from command-line arguments.

        Args:
            args: Parsed command-line arguments

        Returns:
            Instance of WindowTransform
        """
        # Extract arguments
        source_prefix = getattr(args, 'source_prefix', cls.SOURCE_PREFIX)
        dest_prefix = getattr(args, 'dest_prefix', cls.DEST_PREFIX)

        return cls(
            source_prefix=source_prefix,
            destination_prefix=dest_prefix,
            window_size_sec=args.window_size,
            window_step_sec=args.window_step,
            s3_bucket=args.s3_bucket,
            verbose=args.verbose,
            log_file=args.log_file,
            dry_run=args.dry_run,  # Passing to base class only
            keep_local=args.keep_local  # Passing to base class only
        )


# Entry point for running the transform from the command line
if __name__ == "__main__":
    WindowTransform.run_from_command_line()
