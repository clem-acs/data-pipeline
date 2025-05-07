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

# Import neural processing modules
try:
    # First try relative import for package structure
    from .neural_processing.eeg_preprocessing import preprocess_eeg
    from .neural_processing.fnirs_preprocessing import preprocess_fnirs
    from .neural_processing.windowing import create_windows
    from .neural_processing.postprocessing import postprocess_windows
    from .neural_processing.window_dataset import WindowDataset
except ImportError:
    try:
        # Fall back to direct import when running as script
        from transforms.neural_processing.eeg_preprocessing import preprocess_eeg
        from transforms.neural_processing.fnirs_preprocessing import preprocess_fnirs
        from transforms.neural_processing.windowing import create_windows
        from transforms.neural_processing.postprocessing import postprocess_windows
        from transforms.neural_processing.window_dataset import WindowDataset
    except ImportError:
        # Last resort: try direct import if neural_processing is in the path
        import sys
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        neural_processing_dir = os.path.join(current_dir, 'neural_processing')
        sys.path.insert(0, neural_processing_dir)
        
        from eeg_preprocessing import preprocess_eeg
        from fnirs_preprocessing import preprocess_fnirs
        from windowing import create_windows
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
            windows = create_windows(
                processed_eeg, processed_fnirs,
                eeg_timestamps, fnirs_timestamps
            )
            
            # Check if windows were created
            if not windows or len(windows) == 0:
                self.logger.warning(f"No windows created for session {session_id}")
                return {
                    "status": "skipped",
                    "metadata": {"session_id": session_id, "reason": "No windows created"},
                    "files_to_copy": [],
                    "files_to_upload": []
                }
            
            # 5. Apply post-processing to windows
            eeg_processed, fnirs_processed, metadata_processed = postprocess_windows(
                [w['eeg'] for w in windows],
                [w['fnirs'] for w in windows],
                [w['metadata'] for w in windows]
            )
            
            # 6. Create output file
            output_filename = f"{session_id}_windowed.h5"
            local_output_path = session.create_upload_file(output_filename)
            
            # 7. Save windows to output file
            with h5py.File(local_output_path, 'w') as f:
                # Create file-level attributes
                f.attrs['source_file'] = h5_key
                f.attrs['window_size_ms'] = 210
                f.attrs['num_windows'] = len(windows)
                f.attrs['created_at'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
                
                # Store alignment and missing frame information
                if windows and len(windows) > 0:
                    # Extract metadata from the first window
                    first_window_meta = windows[0]['metadata']
                    
                    # Store common time range information
                    if 'alignment_start_time' in first_window_meta and 'alignment_end_time' in first_window_meta:
                        start_time = float(first_window_meta['alignment_start_time'])
                        end_time = float(first_window_meta['alignment_end_time'])
                        common_duration_ms = (end_time - start_time) * 1000
                        
                        f.attrs['alignment_start_time'] = start_time
                        f.attrs['alignment_end_time'] = end_time
                        f.attrs['common_time_range_ms'] = common_duration_ms
                        
                        # Store info about trimmed data amounts
                        if 'eeg_trimmed_start_ms' in first_window_meta and 'eeg_trimmed_end_ms' in first_window_meta:
                            f.attrs['eeg_trimmed_start_ms'] = first_window_meta['eeg_trimmed_start_ms']
                            f.attrs['eeg_trimmed_end_ms'] = first_window_meta['eeg_trimmed_end_ms']
                            f.attrs['eeg_trimmed_percent'] = first_window_meta['eeg_trimmed_percent']
                        
                        if 'fnirs_trimmed_start_ms' in first_window_meta and 'fnirs_trimmed_end_ms' in first_window_meta:
                            f.attrs['fnirs_trimmed_start_ms'] = first_window_meta['fnirs_trimmed_start_ms']
                            f.attrs['fnirs_trimmed_end_ms'] = first_window_meta['fnirs_trimmed_end_ms']
                            f.attrs['fnirs_trimmed_percent'] = first_window_meta['fnirs_trimmed_percent']
                        
                        self.logger.info(f"Common time range: {start_time:.3f}s to {end_time:.3f}s ({common_duration_ms:.1f}ms)")
                    
                    # Count windows with missing fNIRS data
                    missing_fnirs_count = sum(1 for w in windows if w['metadata'].get('missing_fnirs', False))
                    if missing_fnirs_count > 0:
                        missing_percentage = round(100 * missing_fnirs_count / len(windows), 2)
                        f.attrs['missing_fnirs_count'] = missing_fnirs_count
                        f.attrs['missing_fnirs_percentage'] = missing_percentage
                        self.logger.info(f"Stored information about {missing_fnirs_count} windows ({missing_percentage}%) with missing fNIRS data")
                
                # Create data groups
                eeg_group = f.create_group('eeg_windows')
                fnirs_group = f.create_group('fnirs_windows')
                metadata_group = f.create_group('metadata')
                
                # Store original data
                if processed_eeg is not None:
                    f.create_dataset('eeg_data', data=processed_eeg, compression="gzip", compression_opts=4)
                if processed_fnirs is not None:
                    f.create_dataset('fnirs_data', data=processed_fnirs, compression="gzip", compression_opts=4)
                if eeg_timestamps is not None:
                    f.create_dataset('eeg_timestamps', data=eeg_timestamps, compression="gzip", compression_opts=4)
                if fnirs_timestamps is not None:
                    f.create_dataset('fnirs_timestamps', data=fnirs_timestamps, compression="gzip", compression_opts=4)
                
                # Store each window
                for i in range(len(windows)):
                    # Store processed data
                    eeg_group.create_dataset(f"window_{i}",
                                          data=eeg_processed[i] if isinstance(eeg_processed[i], np.ndarray) else eeg_processed[i].numpy(),
                                          compression="gzip", compression_opts=4)
                    
                    fnirs_group.create_dataset(f"window_{i}",
                                            data=fnirs_processed[i] if isinstance(fnirs_processed[i], np.ndarray) else fnirs_processed[i].numpy(),
                                            compression="gzip", compression_opts=4)
                    
                    # Convert any numpy arrays in metadata to native Python types
                    metadata_copy = {}
                    for k, v in metadata_processed[i].items():
                        if isinstance(v, np.ndarray):
                            metadata_copy[k] = v.tolist()  # Convert numpy array to list
                        elif isinstance(v, np.number):
                            metadata_copy[k] = v.item()    # Convert numpy scalar to native Python type
                        else:
                            metadata_copy[k] = v
                    
                    # Store metadata as JSON string
                    metadata_json = json.dumps(metadata_copy)
                    metadata_group.create_dataset(f"window_{i}", data=metadata_json)
            
            # 8. Create result metadata
            result_metadata = {
                "session_id": session_id,
                "window_size_ms": 210,
                "num_windows": len(windows),
                "has_eeg": metadata["has_eeg"],
                "has_fnirs": metadata["has_fnirs"],
                "processed_at": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
            }
            
            # Add alignment and gap information to metadata
            if windows and len(windows) > 0:
                first_window_meta = windows[0]['metadata']
                
                # Add time range information to DynamoDB metadata
                if 'alignment_start_time' in first_window_meta and 'alignment_end_time' in first_window_meta:
                    start_time = float(first_window_meta['alignment_start_time'])
                    end_time = float(first_window_meta['alignment_end_time'])
                    common_duration_ms = (end_time - start_time) * 1000
                    
                    result_metadata['alignment_start_time'] = start_time
                    result_metadata['alignment_end_time'] = end_time
                    result_metadata['common_time_range_ms'] = common_duration_ms
                    
                    # Also add trim percentages to DynamoDB metadata
                    if 'eeg_trimmed_percent' in first_window_meta:
                        result_metadata['eeg_trimmed_percent'] = float(first_window_meta['eeg_trimmed_percent'])
                    
                    if 'fnirs_trimmed_percent' in first_window_meta:
                        result_metadata['fnirs_trimmed_percent'] = float(first_window_meta['fnirs_trimmed_percent'])
                
                # Count windows with missing fNIRS frames
                missing_fnirs_count = sum(1 for w in windows if w['metadata'].get('missing_fnirs', False))
                if missing_fnirs_count > 0:
                    result_metadata['missing_fnirs_count'] = missing_fnirs_count
                    result_metadata['missing_fnirs_percentage'] = round(100 * missing_fnirs_count / len(windows), 2)
            
            if metadata["has_eeg"]:
                result_metadata["eeg_channels"] = processed_eeg.shape[1] if len(processed_eeg.shape) > 1 else 0
            
            if metadata["has_fnirs"]:
                result_metadata["fnirs_channels"] = processed_fnirs.shape[1] if len(processed_fnirs.shape) > 1 else 0
            
            # Define the destination key
            dest_key = f"{self.destination_prefix}{output_filename}"
            
            return {
                "status": "success",
                "metadata": result_metadata,
                "files_to_copy": [],
                "files_to_upload": [(local_output_path, dest_key)]
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
