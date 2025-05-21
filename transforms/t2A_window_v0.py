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
import xarray as xr
from typing import Dict, Any, List, Optional

# Import base transform
from base_transform import BaseTransform, Session

# Import neural processing modules

# Define path to neural_processing directory relative to this script
# This assumes 'neural_processing' is a subdirectory where t2A_window_v0.py is located,
# or that this path correctly points to where layout.json can be found.
NEURAL_PROCESSING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'neural_processing')

# Import neural processing modules using absolute paths from the project root
# This assumes 'data-pipeline' is the top-level package in PYTHONPATH
from neural_processing.eeg_preprocessing import preprocess_eeg
from neural_processing.fnirs_preprocessing import preprocess_fnirs
from neural_processing.windowing import create_windows, create_windows_eeg_only, create_windows_fnirs_only
from neural_processing.normalize import normalize_window_dataset


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

    def __init__(self, **kwargs):
        """
        Initialize the neural windowing transform.

        Args:
            **kwargs: Additional arguments for BaseTransform
        """
        # Set default transform info if not provided
        transform_id = kwargs.pop('transform_id', 't2A_window_v0')
        script_id = kwargs.pop('script_id', '2A')
        script_name = kwargs.pop('script_name', 'window')
        script_version = kwargs.pop('script_version', 'v0')

        # Call parent constructor
        super().__init__(
            transform_id=transform_id,
            script_id=script_id,
            script_name=script_name,
            script_version=script_version,
            **kwargs
        )

        self.logger.info(f"Neural windowing transform initialized with:")
        self.logger.info(f"  Window size: 210ms (1 fNIRS frame, 7 EEG frames)")

    def process_session(self, session: Session) -> Dict:
        """Process a single session.

        This implementation:
        1. Finds the curated H5 file for the session
        2. Extracts and preprocesses EEG and fNIRS data
        3. Creates time-aligned windows
        4. Stores the windowed data

        Args:
            session: Session object

        Returns:
            Dict with processing results
        """
        session_id = session.session_id
        self.logger.info(f"Processing session: {session_id}")

        # Look for H5 file in the source prefix
        h5_key = f"{self.source_prefix}{session_id}.h5"

        # Check if the file exists
        try:
            self.s3.head_object(Bucket=self.s3_bucket, Key=h5_key)
            self.logger.info(f"Found H5 file: {h5_key}")
        except Exception as e:
            self.logger.error(f"No H5 file found for session {session_id}: {e}")
            return {
                "status": "failed",
                "error_details": f"No H5 file found for session {session_id}",
                "metadata": {"session_id": session_id},
                "files_to_copy": [],
                "files_to_upload": []
            }

        # Download the H5 file
        local_h5_path = session.download_file(h5_key)

        try:
            # 1. Extract data from H5 file
            with h5py.File(local_h5_path, 'r') as f:
                eeg_data, eeg_timestamps, fnirs_data, fnirs_timestamps = self.extract_data_from_h5(f)

            # 2. Create metadata
            metadata = {
                "session_id": session_id,
                "window_size_ms": 210,
                "has_eeg": eeg_data is not None and eeg_timestamps is not None,
                "has_fnirs": fnirs_data is not None and fnirs_timestamps is not None
            }

            # At least one modality must be available
            if not metadata["has_eeg"] and not metadata["has_fnirs"]:
                self.logger.error("No valid data found in the H5 file")
                return {
                    "status": "skipped",
                    "metadata": {"session_id": session_id, "reason": "No valid EEG or fNIRS data"},
                    "files_to_copy": [],
                    "files_to_upload": []
                }

            # 3. Preprocess EEG and fNIRS data
            processed_eeg, processed_fnirs = self.preprocess_data(
                eeg_data, eeg_timestamps, fnirs_data, fnirs_timestamps, metadata
            )

            # 4. Create explicit windows
            # 4. Create explicit windows
            # Determine data presence after preprocessing for windowing
            current_has_eeg = processed_eeg is not None and len(processed_eeg) > 0 and \
                              eeg_timestamps is not None and len(eeg_timestamps) > 0
            current_has_fnirs = processed_fnirs is not None and len(processed_fnirs) > 0 and \
                                fnirs_timestamps is not None and len(fnirs_timestamps) > 0

            self.logger.info(f"Data presence for windowing: EEG={current_has_eeg}, fNIRS={current_has_fnirs}")

            dataset_instance = None
            summary_meta = {} # Initialize to empty dict

            if current_has_eeg and current_has_fnirs:
                self.logger.info("Calling create_windows (EEG and fNIRS)...")
                dataset_instance, summary_meta = create_windows(
                    eeg_data=processed_eeg,
                    fnirs_data=processed_fnirs,
                    eeg_timestamps=eeg_timestamps,
                    fnirs_timestamps=fnirs_timestamps,
                    return_torch_tensors=False # We want NumPy arrays for HDF5 saving
                )
            elif current_has_eeg:
                self.logger.info("Calling create_windows_eeg_only...")
                dataset_instance, summary_meta = create_windows_eeg_only(
                    eeg_data=processed_eeg,
                    eeg_timestamps=eeg_timestamps,
                    return_torch_tensors=False
                )
            elif current_has_fnirs:
                self.logger.info("Calling create_windows_fnirs_only...")
                dataset_instance, summary_meta = create_windows_fnirs_only(
                    fnirs_data=processed_fnirs,
                    fnirs_timestamps=fnirs_timestamps,
                    return_torch_tensors=False
                )
            else:
                self.logger.warning("No EEG or fNIRS data available after preprocessing for windowing. Skipping window creation.")
                summary_meta = {
                    "status": "skipped_no_data_for_windowing",
                    "error_details": "No EEG or fNIRS data available after preprocessing for windowing.",
                    "percent_eeg_trimmed": 100.0,
                    "percent_fnirs_trimmed": 100.0,
                    "total_master_windows_in_input_arrays": 0
                }

            self.logger.info(f"Windowing function returned: dataset_instance is {type(dataset_instance)}, "
                             f"summary_meta keys: {list(summary_meta.keys()) if summary_meta and isinstance(summary_meta, dict) else 'None'}")

            # Ensure summary_meta is a dict for downstream processing
            if not isinstance(summary_meta, dict):
                self.logger.warning(f"summary_meta was not a dict ({type(summary_meta)}), re-initializing to empty dict.")
                summary_meta = {}

            # Check if windowing reported a critical error that should halt processing for this session
            # Exclude "skipped_no_data_for_windowing" as that's a valid path to 0 windows.
            if summary_meta.get("error") and summary_meta.get("status") != "skipped_no_data_for_windowing":
                 self.logger.error(f"Windowing function reported a critical error: {summary_meta.get('error')}")
                 # Using original metadata dict from earlier in the function
                 error_return_metadata = {**metadata, **summary_meta} # Merge session metadata with error summary
                 return {
                     "status": "failed",
                     "error_details": f"Windowing error: {summary_meta.get('error', 'Unknown windowing failure')}",
                     "metadata": error_return_metadata,
                     "files_to_copy": [],
                     "files_to_upload": []
                 }

            # 5. Normalize data using the normalize_window_dataset function
            retained_fnirs_indices = None
            normalization_success = False

            if isinstance(summary_meta, dict) and 'retained_fnirs_indices' in summary_meta:
                retained_fnirs_indices = summary_meta.get('retained_fnirs_indices')
                self.logger.info(f"Found retained_fnirs_indices in summary_meta with {len(retained_fnirs_indices) if retained_fnirs_indices else 0} indices")

            if dataset_instance is not None:
                self.logger.info("Normalizing window dataset...")
                # No exception handling - let errors propagate up to fail the transform
                normalized_dataset = normalize_window_dataset(dataset_instance, retained_fnirs_indices)
                self.logger.info("Window dataset normalized successfully")
                dataset_instance = normalized_dataset
                normalization_success = True
            else:
                self.logger.info("Skipping normalization as dataset_instance is None")

            num_dataset_windows = len(dataset_instance) if dataset_instance is not None else 0

            # 6. & 7. Create and save xarray dataset directly to S3
            zarr_key = f"{self.destination_prefix}{session_id}_windowed.zarr"
            
            if dataset_instance is not None:
                self.logger.info(f"Creating xarray dataset and saving directly to S3")
                
                # Create data arrays
                n_windows = len(dataset_instance)
                timestamps = np.zeros(n_windows, dtype=np.float64)
                eeg_valid = np.zeros(n_windows, dtype=bool)
                fnirs_valid = np.zeros(n_windows, dtype=bool)
                real_eeg_ts = np.zeros(n_windows, dtype=np.float64)
                real_fnirs_ts = np.zeros(n_windows, dtype=np.float64)
                original_array_idx = np.zeros(n_windows, dtype=np.int32)
                dataset_idx = np.zeros(n_windows, dtype=np.int32)
                
                # Get shapes
                first_window = dataset_instance[0]
                eeg_shape = first_window['eeg'].shape
                fnirs_shape = first_window['fnirs'].shape
                
                # Initialize data arrays
                eeg_data = np.zeros((n_windows, *eeg_shape), dtype=np.float32)
                fnirs_data = np.zeros((n_windows, *fnirs_shape), dtype=np.float32)
                
                # Fill arrays
                for i in range(n_windows):
                    window = dataset_instance[i]
                    metadata = window['metadata']
                    
                    timestamps[i] = metadata['main_clock_ts']
                    eeg_data[i] = window['eeg']
                    fnirs_data[i] = window['fnirs']
                    
                    eeg_valid[i] = metadata['eeg_valid']
                    fnirs_valid[i] = metadata['fnirs_valid']
                    real_eeg_ts[i] = metadata['real_eeg_start_ts']
                    real_fnirs_ts[i] = metadata['real_fnirs_start_ts']
                    original_array_idx[i] = metadata['original_array_idx']
                    dataset_idx[i] = metadata['dataset_idx']
                
                # Extract master indices
                master_eeg_start = getattr(dataset_instance, 'master_eeg_start_idx', -1)
                master_fnirs_start = getattr(dataset_instance, 'master_fnirs_start_idx', -1)
                master_eeg_end = getattr(dataset_instance, 'master_eeg_end_idx', -1)
                master_fnirs_end = getattr(dataset_instance, 'master_fnirs_end_idx', -1)
                slice_start = getattr(dataset_instance, 'slice_start', -1)
                slice_end = getattr(dataset_instance, 'slice_end', -1)

                # Squeeze the singleton dimension from fnirs_data if present
                fnirs_data = np.squeeze(fnirs_data, axis=-1)

                # Create xarray Dataset with proper dimensions
                eeg_dims = ['time', 'eeg_channel', 'eeg_sample']
                fnirs_dims = ['time', 'fnirs_channel']
                
                ds = xr.Dataset(
                    data_vars={
                        'eeg': (eeg_dims, eeg_data),
                        'fnirs': (fnirs_dims, fnirs_data),
                        'eeg_valid': (['time'], eeg_valid),
                        'fnirs_valid': (['time'], fnirs_valid),
                        'real_eeg_ts': (['time'], real_eeg_ts),
                        'real_fnirs_ts': (['time'], real_fnirs_ts),
                        'original_array_idx': (['time'], original_array_idx),
                        'dataset_idx': (['time'], dataset_idx),
                    },
                    coords={
                        'time': timestamps,
                    },
                    attrs={
                        'master_eeg_start_idx': master_eeg_start,
                        'master_fnirs_start_idx': master_fnirs_start,
                        'master_eeg_end_idx': master_eeg_end,
                        'master_fnirs_end_idx': master_fnirs_end,
                        'slice_start': slice_start,
                        'slice_end': slice_end
                    }
                )
                
                # Set chunking (100 windows per chunk)
                chunks = {'time': min(100, n_windows)}
                
                # Apply chunking (will be used by save_dataset_to_s3_zarr)
                ds = ds.chunk(chunks)

                # Convert xarray Dataset to dictionary structure for Zarr 3
                zarr_tree = {}
                attrs = dict(ds.attrs)
                
                # Add storage format attribute
                attrs['storage_format'] = 'zarr3'
                
                # Process data variables and coordinates
                for name, array in {**ds.data_vars, **ds.coords}.items():
                    zarr_tree[name] = array.values
                
                # Save using Zarr 3 method
                self.save_zarr_dict_to_s3(zarr_key, zarr_tree, attrs)
                self.logger.info(f"Successfully saved dataset directly to S3 using Zarr 3 format")
                
                # Add zarr store to results for proper handling
                zarr_stores = [zarr_key]
            else:
                self.logger.warning("No dataset to save to Zarr")
                zarr_stores = []

            # 8. Create result metadata for DynamoDB
            result_metadata = {
                "session_id": session_id,
                "original_window_size_ms": metadata.get("window_size_ms", 210),
                "num_dataset_windows": num_dataset_windows,
                "has_eeg_initial": metadata.get("has_eeg", False), # Based on raw data from H5
                "has_fnirs_initial": metadata.get("has_fnirs", False), # Based on raw data from H5
                "has_eeg_processed": current_has_eeg, # Reflects data presence post-preprocessing
                "has_fnirs_processed": current_has_fnirs, # Reflects data presence post-preprocessing
                "normalization_applied": normalization_success,
                "storage_format": "zarr_xarray",
                "processed_at": time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime()), # Use gmtime for UTC
            }
            # Add relevant details from summary_meta if it exists and is a dict
            if isinstance(summary_meta, dict):
                keys_from_summary = [
                    'percent_eeg_trimmed', 'percent_fnirs_trimmed',
                    'percent_fnirs_missing_in_span',
                    'total_master_windows_in_input_arrays',
                    'status', # e.g., "skipped_no_data_for_windowing"
                    'error_details' # if any error was reported by windowing and wasn't critical
                ]
                for key in keys_from_summary:
                    if key in summary_meta:
                        # Prefix to avoid clashes and indicate source
                        dynamo_key = f"windowing_{key.lower().replace(' ', '_')}"
                        value_to_store = summary_meta[key]
                        # Ensure value is DynamoDB compatible (string, number, bool, null, list, map)
                        if not isinstance(value_to_store, (str, int, float, bool, type(None))):
                            value_to_store = str(value_to_store) # Convert complex types to string
                        result_metadata[dynamo_key] = value_to_store

            return {
                "status": "success",
                "metadata": result_metadata,
                "files_to_copy": [],
                "files_to_upload": [],
                "zarr_stores": zarr_stores
            }

        except Exception as e:
            self.logger.error(f"Error processing session {session_id}: {e}", exc_info=True)
            error_metadata = {"session_id": session_id}
            if 'summary_meta' in locals() and summary_meta:
                 error_metadata.update({k: v for k, v in summary_meta.items() if isinstance(v, (int, float, str, bool, type(None)))})
            return {
                "status": "failed",
                "error_details": str(e),
                "metadata": error_metadata,
                "files_to_copy": [],
                "files_to_upload": [],
                "zarr_stores": []
            }

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
            Tuple of (processed_eeg, processed_fnirs)
        """
        has_eeg = eeg_data is not None and eeg_timestamps is not None
        has_fnirs = fnirs_data is not None and fnirs_timestamps is not None

        # Process EEG if available
        processed_eeg = None
        if has_eeg:
            self.logger.info("Preprocessing EEG data")
            processed_eeg, eeg_preprocessing_metadata = preprocess_eeg(eeg_data, metadata)

            self.logger.debug(f"EEG data preprocessed to shape {processed_eeg.shape}")

            metadata.update(eeg_preprocessing_metadata)
        else:
            # Create placeholders - using 3D shape for consistency
            processed_eeg = np.zeros((1, 1, 1), dtype=np.float32)
            metadata['eeg_preprocessing_skipped'] = True
            metadata['eeg_channels'] = 0

        # Process fNIRS if available
        processed_fnirs = None
        if has_fnirs:
            self.logger.info("Preprocessing fNIRS data with spatial filtering")

            # Load layout data if available
            layout_json_path = os.path.join(NEURAL_PROCESSING_DIR, 'layout.json')
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
            excluded_valid = fnirs_preprocessing_metadata.get('excluded_channels_with_valid_data', 0)
            all_invalid = fnirs_preprocessing_metadata.get('all_excluded_channels_invalid', False)
            valid_percentage = fnirs_preprocessing_metadata.get('valid_channel_percentage', 0)

            # Detailed logging of channel quality
            self.logger.info(f"fNIRS channel quality: {included_valid} of {used_channels} included channels ({valid_percentage:.1f}%) contain valid data")

            # Add information about excluded channels
            if excluded_valid > 0:
                self.logger.info(f"fNIRS spatial filtering: {excluded_valid} channels with valid data were excluded based on distance constraints")
            elif all_invalid:
                self.logger.info(f"fNIRS spatial filtering: All excluded channels contained only invalid data")

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

        return processed_eeg, processed_fnirs

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

        kwargs = {
            'source_prefix': source_prefix,
            'destination_prefix': dest_prefix,
            's3_bucket': args.s3_bucket,
            'verbose': args.verbose,
            'log_file': args.log_file,
            'dry_run': args.dry_run
        }

        # Handle keep_local if it exists
        if hasattr(args, 'keep_local'):
            kwargs['keep_local'] = args.keep_local

        return cls(**kwargs)


# Entry point for running the transform from the command line
if __name__ == "__main__":
    WindowTransform.run_from_command_line()
