"""
Audio extraction transform for H5 files in the pipeline.

This transform:
1. Reads curated H5 files from S3 (curated-h5/)
2. Identifies audio data and relevant event timestamps ("recording_start", "recording_stop").
3. (Future) Extracts audio segments based on these events and saves them as WAV files.
4. This initial version logs the H5 attributes and dataset structures.
5. Saves a JSON report to processed/t2F_audio_extract/ prefix in S3.
6. Records metadata in DynamoDB.

This is implemented using the BaseTransform architecture.
"""

import os
import sys
import json
import h5py
import numpy as np
import scipy.io.wavfile
import scipy.stats
from typing import Dict, Any, List, Optional, Tuple, Union

# Import base transform
# Assuming base_transform.py is in the parent directory of transforms/
# or that it's installed/available in PYTHONPATH correctly.
# For direct script running where data-pipeline is in PYTHONPATH:
try:
    # Attempt import assuming base_transform is accessible directly
    # This works if data-pipeline is in PYTHONPATH or installed
    from base_transform import BaseTransform, Session
except ImportError:
    # Fallback: Add parent directory to sys.path if running as script
    # This helps find base_transform.py if it's in the data-pipeline root
    # and this script is run directly from data-pipeline/transforms
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    try:
        from base_transform import BaseTransform, Session
    except ImportError as e:
        print(f"Error importing BaseTransform: {e}. Ensure base_transform.py is accessible.", file=sys.stderr)
        sys.exit(1)


class AudioExtractTransform(BaseTransform):
    """
    Audio extraction transform for H5 files.

    This transform will eventually extract audio segments.
    This initial version focuses on logging relevant H5 structures.
    """

    SOURCE_PREFIX = 'curated-h5/'
    # Destination for both reports and extracted audio files
    DEST_PREFIX = 'processed/t2F_audio_extract/'

    def __init__(self, min_segment_duration=0.5, normalize_audio=True, force_sample_rate=None, 
                stats_window_size=10.0, outlier_threshold=3.0, **kwargs):
        """
        Initialize the audio extraction transform.

        Args:
            min_segment_duration: Minimum duration in seconds for extracted audio segments
            normalize_audio: Whether to normalize audio data to range [-1.0, 1.0]
            force_sample_rate: Force specific sample rate instead of auto-detecting
            stats_window_size: Window size in seconds for audio statistics
            outlier_threshold: Z-score threshold for identifying outlier windows
            **kwargs: Additional arguments for BaseTransform
        """
        transform_id = kwargs.pop('transform_id', 'audio_extract_v0')
        script_id = kwargs.pop('script_id', 't2F')
        script_name = kwargs.pop('script_name', 'audio_extract_h5')
        script_version = kwargs.pop('script_version', 'v0')

        super().__init__(
            transform_id=transform_id,
            script_id=script_id,
            script_name=script_name,
            script_version=script_version,
            **kwargs
        )
        
        self.min_segment_duration = min_segment_duration
        self.normalize_audio = normalize_audio
        self.force_sample_rate = force_sample_rate
        self.stats_window_size = stats_window_size
        self.outlier_threshold = outlier_threshold
        
        self.logger.info(f"AudioExtractTransform initialized with:")
        self.logger.info(f"  Minimum segment duration: {self.min_segment_duration} seconds")
        self.logger.info(f"  Normalize audio: {self.normalize_audio}")
        self.logger.info(f"  Force sample rate: {self.force_sample_rate or 'auto-detect'}")
        self.logger.info(f"  Statistics window size: {self.stats_window_size} seconds")
        self.logger.info(f"  Outlier threshold: {self.outlier_threshold} sigma")

    def _log_dataset_info(self, h5_file: h5py.File, dataset_path: str) -> Optional[Dict[str, Any]]:
        """Helper function to log info about a dataset and its attributes."""
        if dataset_path in h5_file:
            dataset = h5_file[dataset_path]
            info = {
                "path": dataset_path,
                "shape": dataset.shape,
                "dtype": str(dataset.dtype),
                "attributes": {}
            }
            for attr_name, attr_value in dataset.attrs.items():
                # Convert numpy types to standard Python types for JSON serialization
                if isinstance(attr_value, (np.ndarray, np.generic)):
                    attr_value = attr_value.tolist()
                # Decode bytes to string if possible
                if isinstance(attr_value, bytes):
                    try:
                        attr_value = attr_value.decode('utf-8')
                    except UnicodeDecodeError:
                        # Keep as string representation if decoding fails
                        attr_value = str(attr_value)
                info["attributes"][attr_name] = attr_value
            self.logger.info(f"Found dataset: {dataset_path}, Shape: {dataset.shape}, Dtype: {dataset.dtype}, Attrs: {info['attributes']}")
            return info
        else:
            self.logger.warning(f"Dataset not found: {dataset_path}")
            return None
            
    def _extract_timestamps(self, h5_file: h5py.File, dataset_path: str, limit: Optional[int] = None) -> Optional[List]:
        """Extract timestamp data from a dataset, with optional limit on number of rows."""
        try:
            if dataset_path in h5_file:
                dataset = h5_file[dataset_path]
                # If limit is specified and dataset has more rows than limit, only take the first 'limit' rows
                if limit is not None and dataset.shape[0] > limit:
                    data = dataset[:limit]
                else:
                    data = dataset[:]
                # Convert numpy array to Python list for JSON serialization
                return data.tolist()
            else:
                self.logger.warning(f"Dataset not found for timestamp extraction: {dataset_path}")
                return None
        except Exception as e:
            self.logger.error(f"Error extracting timestamps from {dataset_path}: {e}")
            return None
            
    def _estimate_sample_rate(self, timestamps: List[List[float]], samples_per_chunk: int = 4096) -> int:
        """Estimate audio sample rate from timestamp data.
        
        Args:
            timestamps: List of [start, end] timestamp pairs
            samples_per_chunk: Number of samples in each audio chunk
            
        Returns:
            Estimated sample rate in Hz
        """
        # If a forced sample rate is set, use that
        if self.force_sample_rate is not None:
            self.logger.info(f"Using forced sample rate: {self.force_sample_rate} Hz")
            return self.force_sample_rate
            
        # Otherwise estimate from timestamps
        if not timestamps or len(timestamps) < 2:
            # Default to 16000 Hz if we don't have enough timestamps
            self.logger.warning("Not enough timestamps to estimate sample rate, using default 16000 Hz")
            return 16000
            
        # Use client-side timestamps (index 1) for calculation
        # Calculate time differences between consecutive chunks
        time_diffs_ms = []
        for i in range(1, len(timestamps)):
            time_diff = timestamps[i][1] - timestamps[i-1][1]
            if time_diff > 0:  # Ensure valid time difference
                time_diffs_ms.append(time_diff)
                
        if not time_diffs_ms:
            self.logger.warning("Unable to calculate time differences, using default 16000 Hz")
            return 16000
            
        # Calculate average time between chunks in seconds
        avg_chunk_time_sec = np.mean(time_diffs_ms) / 1000.0
        
        # Calculate samples per second based on chunk size and time
        estimated_rate = int(round(samples_per_chunk / avg_chunk_time_sec))
        
        # Common sample rates are 8000, 16000, 22050, 44100, 48000
        # Round to the nearest common rate
        common_rates = [8000, 16000, 22050, 44100, 48000]
        closest_rate = min(common_rates, key=lambda x: abs(x - estimated_rate))
        
        self.logger.info(f"Estimated sample rate: {estimated_rate} Hz, using closest common rate: {closest_rate} Hz")
        return closest_rate
        
    def _extract_audio_segments(self, audio_data, audio_timestamps, start_events, stop_events, sample_rate) -> List[Tuple[np.ndarray, str]]:
        """Extract audio segments between recording_start and recording_stop events.
        
        Args:
            audio_data: Full audio data array
            audio_timestamps: List of [start, end] timestamp pairs for audio chunks
            start_events: List of recording_start timestamps
            stop_events: List of recording_stop timestamps
            sample_rate: Estimated sample rate
            
        Returns:
            List of tuples (audio_segment, segment_name)
        """
        segments = []
        
        # Validate inputs
        if audio_data is None or len(audio_data) == 0:
            self.logger.error("No audio data provided for segment extraction")
            return segments
            
        if not audio_timestamps or len(audio_timestamps) == 0:
            self.logger.error("No audio timestamps provided for segment extraction")
            return segments
            
        if not start_events or not stop_events:
            self.logger.error("Missing recording start or stop events")
            return segments
        
        try:
            # Client-side timestamps are at index 1
            # Convert audio timestamps to a numpy array for easier searching
            # Handle different possible formats of audio_timestamps
            if isinstance(audio_timestamps, list) and len(audio_timestamps) > 0:
                if isinstance(audio_timestamps[0], list) and len(audio_timestamps[0]) > 1:
                    # This is the format from _extract_timestamps: [[start1, end1], [start2, end2], ...]
                    client_audio_ts = np.array([ts[1] for ts in audio_timestamps])
                elif len(np.array(audio_timestamps).shape) == 2 and np.array(audio_timestamps).shape[1] == 2:
                    # This is similar but already as numpy array
                    client_audio_ts = np.array(audio_timestamps)[:, 1]
                else:
                    # Just use the timestamps as is if format is unexpected
                    client_audio_ts = np.array(audio_timestamps)
            else:
                self.logger.error("Audio timestamps in unexpected format")
                return segments
            
            # Process each recording session (start/stop pair)
            # Skip the first stop event if it occurs before any start event
            start_idx = 0
            if stop_events and start_events and stop_events[0][1] < start_events[0][1]:
                start_idx = 1
                self.logger.info("First stop event occurs before first start event, adjusting pairing")
            
        except Exception as e:
            self.logger.error(f"Error setting up audio segment extraction: {e}")
            return segments
            
        for i in range(len(start_events)):
            # Get the corresponding stop event
            if i + start_idx >= len(stop_events):
                self.logger.warning(f"No matching stop event for start event {i+1}, skipping")
                continue
                
            # Get client-side timestamps
            try:
                start_ts = start_events[i][1]
                stop_ts = stop_events[i + start_idx][1]
            except (IndexError, TypeError) as e:
                self.logger.error(f"Error accessing timestamps for segment {i+1}: {e}")
                continue
            
            # Calculate segment duration in seconds
            try:
                duration_sec = (stop_ts - start_ts) / 1000.0
            except TypeError as e:
                self.logger.error(f"Error calculating duration for segment {i+1}: {e}")
                continue
            
            # Skip segments that are too short
            if duration_sec < self.min_segment_duration:
                self.logger.warning(f"Segment {i+1} duration ({duration_sec:.2f}s) is below minimum ({self.min_segment_duration}s), skipping")
                continue
                
            # Find audio chunks that fall within this range
            try:
                # Find the first chunk that starts after or at the start_ts
                chunk_start_idx = np.searchsorted(client_audio_ts, start_ts)
                # Find the last chunk that starts before or at the stop_ts
                chunk_stop_idx = np.searchsorted(client_audio_ts, stop_ts, side='right') - 1
                
                if chunk_start_idx >= len(audio_data) or chunk_stop_idx < 0 or chunk_start_idx > chunk_stop_idx:
                    self.logger.warning(f"Invalid audio chunk range for segment {i+1}: {chunk_start_idx} to {chunk_stop_idx}")
                    continue
                    
                # Extract the audio chunks and concatenate them
                segment_data = audio_data[chunk_start_idx:chunk_stop_idx+1].flatten()
                
                # Calculate actual segment duration based on samples
                actual_duration = len(segment_data) / sample_rate
                
                # Create segment name with index and timestamp info
                segment_name = f"segment_{i+1:02d}_{int(start_ts):013d}_{int(stop_ts):013d}"
                
                segments.append((segment_data, segment_name))
                self.logger.info(f"Extracted audio segment {i+1}: {len(segment_data)} samples, {actual_duration:.2f} seconds")
            except Exception as e:
                self.logger.error(f"Error extracting segment {i+1}: {e}")
                continue
            
        if not segments:
            self.logger.warning("No valid audio segments were extracted")
            
        return segments
        
    def _calculate_audio_stats(self, audio_data: np.ndarray, sample_rate: int, window_sec: float = 10.0, overlap: float = 0.0) -> Dict:
        """Calculate audio statistics in sliding windows.
        
        Args:
            audio_data: Complete audio data array 
            sample_rate: Sample rate in Hz
            window_sec: Window size in seconds
            overlap: Overlap between windows (0.0-1.0)
            
        Returns:
            Dictionary of window statistics
        """
        if audio_data.ndim > 1:
            # Flatten multi-dimensional array if needed
            flattened_audio = audio_data.flatten()
        else:
            flattened_audio = audio_data
            
        samples_per_window = int(window_sec * sample_rate)
        step_size = int(samples_per_window * (1.0 - overlap))
        
        window_stats = []
        
        # Process each window
        for start_idx in range(0, len(flattened_audio) - samples_per_window + 1, step_size):
            end_idx = start_idx + samples_per_window
            window_data = flattened_audio[start_idx:end_idx]
            
            # Calculate statistics
            mean = np.mean(window_data)
            std = np.std(window_data)
            rms = np.sqrt(np.mean(np.square(window_data)))
            max_abs = np.max(np.abs(window_data))
            percentile_95 = np.percentile(np.abs(window_data), 95)
            
            # Calculate time range
            start_time_sec = start_idx / sample_rate
            end_time_sec = end_idx / sample_rate
            
            window_stats.append({
                "window_start_sec": start_time_sec,
                "window_end_sec": end_time_sec,
                "mean": float(mean),
                "std": float(std),
                "rms": float(rms),
                "max_abs": float(max_abs),
                "percentile_95": float(percentile_95)
            })
            
        # Calculate global statistics
        all_means = [w["mean"] for w in window_stats]
        all_stds = [w["std"] for w in window_stats]
        all_rms = [w["rms"] for w in window_stats]
        
        stats_summary = {
            "window_stats": window_stats,
            "global": {
                "mean_of_means": float(np.mean(all_means)),
                "std_of_means": float(np.std(all_means)),
                "mean_of_stds": float(np.mean(all_stds)),
                "std_of_stds": float(np.std(all_stds)),
                "mean_of_rms": float(np.mean(all_rms)),
                "std_of_rms": float(np.std(all_rms)),
            }
        }
        
        return stats_summary
        
    def _identify_outliers(self, stats: Dict, z_threshold: float = 3.0) -> List[Dict]:
        """Identify outlier windows based on z-score.
        
        Args:
            stats: Window statistics from _calculate_audio_stats
            z_threshold: Z-score threshold for outliers
            
        Returns:
            List of outlier windows
        """
        if not stats or "window_stats" not in stats or not stats["window_stats"]:
            return []
            
        window_stats = stats["window_stats"]
        means = np.array([w["mean"] for w in window_stats])
        stds = np.array([w["std"] for w in window_stats])
        rms_values = np.array([w["rms"] for w in window_stats])
        
        # Calculate z-scores
        mean_z_scores = np.abs(scipy.stats.zscore(means))
        std_z_scores = np.abs(scipy.stats.zscore(stds))
        rms_z_scores = np.abs(scipy.stats.zscore(rms_values))
        
        outliers = []
        
        for i, window in enumerate(window_stats):
            # Check if this window is an outlier in any metric
            if (mean_z_scores[i] > z_threshold or 
                std_z_scores[i] > z_threshold or 
                rms_z_scores[i] > z_threshold):
                
                outlier_info = window.copy()
                outlier_info.update({
                    "mean_z_score": float(mean_z_scores[i]),
                    "std_z_score": float(std_z_scores[i]),
                    "rms_z_score": float(rms_z_scores[i]),
                    "is_outlier": True
                })
                outliers.append(outlier_info)
                
        return outliers
        
    def _calculate_segment_stats(self, segments: List[Tuple], sample_rate: int) -> Dict:
        """Calculate statistics for extracted audio segments.
        
        Args:
            segments: List of (audio_segment, segment_name) tuples
            sample_rate: Sample rate in Hz
            
        Returns:
            Dictionary of segment statistics
        """
        segment_stats = []
        
        for segment_data, segment_name in segments:
            # Calculate statistics for this segment
            mean = np.mean(segment_data)
            std = np.std(segment_data)
            rms = np.sqrt(np.mean(np.square(segment_data)))
            max_abs = np.max(np.abs(segment_data))
            duration_sec = len(segment_data) / sample_rate
            
            segment_stats.append({
                "segment_name": segment_name,
                "duration_sec": float(duration_sec),
                "mean": float(mean),
                "std": float(std),
                "rms": float(rms),
                "max_abs": float(max_abs),
                "zero_percentage": float(np.sum(segment_data == 0) / len(segment_data) * 100)
            })
            
        return segment_stats
    
    def _save_wav_file(self, session: Session, audio_data: np.ndarray, sample_rate: int, segment_name: str) -> str:
        """Save audio data as a WAV file.
        
        Args:
            session: Session object to manage file paths
            audio_data: Audio data array
            sample_rate: Sample rate in Hz
            segment_name: Segment name for the filename
            
        Returns:
            Path to the saved WAV file
        """
        # Create WAV filename
        wav_filename = f"{session.session_id}_{segment_name}.wav"
        local_wav_path = session.create_upload_file(wav_filename)
        
        # Normalize audio data to float32 in range [-1.0, 1.0]
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
            
        # Only normalize if configured to do so and the data isn't all zeros
        if self.normalize_audio and np.max(np.abs(audio_data)) > 0:
            audio_data = audio_data / np.max(np.abs(audio_data))
            self.logger.info(f"Normalized audio data for {segment_name}")
        
        try:
            # Save as WAV file
            scipy.io.wavfile.write(local_wav_path, sample_rate, audio_data)
            self.logger.info(f"Saved WAV file: {local_wav_path}")
            return local_wav_path
        except Exception as e:
            self.logger.error(f"Error saving WAV file {local_wav_path}: {e}")
            raise

    def _log_group_attributes(self, h5_file: h5py.File, group_path: str) -> Optional[Dict[str, Any]]:
        """Helper function to log attributes of a group."""
        if group_path in h5_file:
            group = h5_file[group_path]
            attributes = {}
            for attr_name, attr_value in group.attrs.items():
                if isinstance(attr_value, (np.ndarray, np.generic)):
                    attr_value = attr_value.tolist()
                if isinstance(attr_value, bytes):
                    try:
                        attr_value = attr_value.decode('utf-8')
                    except UnicodeDecodeError:
                        attr_value = str(attr_value)
                attributes[attr_name] = attr_value
            self.logger.info(f"Found group: {group_path}, Attributes: {attributes}")
            # Specifically look for sampling rate common names
            for sr_key in ['sampling_rate', 'sample_rate', 'fs', 'samplerate']:
                if sr_key in attributes:
                    self.logger.info(f"Potential sampling rate found in {group_path} attributes: {sr_key} = {attributes[sr_key]}")
            return attributes
        else:
            self.logger.warning(f"Group not found: {group_path}")
            return None
            
    def _extract_language_speech(self, h5_file: h5py.File) -> Optional[List[Dict[str, Any]]]:
        """Extract speech transcription data from the /language/S group.
        
        Args:
            h5_file: Open H5 file object
            
        Returns:
            List of speech items with word, timestamps, and metadata
        """
        speech_items = []
        try:
            if "/language/S/words" in h5_file:
                words_dataset = h5_file["/language/S/words"]
                self.logger.info(f"Found speech dataset: shape {words_dataset.shape}, dtype {words_dataset.dtype}")
                
                # Get all speech items
                for item in words_dataset[:]:
                    # Extract fields from the compound dataset
                    speech_item = {}
                    for field_name in words_dataset.dtype.names:
                        value = item[field_name]
                        if isinstance(value, bytes):
                            try:
                                value = value.decode('utf-8')
                            except UnicodeDecodeError:
                                value = str(value)
                        speech_item[field_name] = value
                    speech_items.append(speech_item)
                
                self.logger.info(f"Extracted {len(speech_items)} speech items from /language/S/words")
                return speech_items
            else:
                self.logger.warning("Speech dataset (/language/S/words) not found")
                return None
        except Exception as e:
            self.logger.error(f"Error extracting speech data: {e}")
            return None
            
    def _extract_element_events(self, h5_file: h5py.File, event_type: str) -> Tuple[Optional[List], Optional[List]]:
        """Extract element event data (element_sent or element_replied).
        
        Args:
            h5_file: Open H5 file object
            event_type: Type of element event ('element_sent' or 'element_replied')
            
        Returns:
            Tuple of (timestamps, data) lists if successful, (None, None) otherwise
        """
        timestamps = None
        data = None
        event_path = f"/events/{event_type}"
        
        try:
            # Check if this event type exists
            if event_path not in h5_file:
                self.logger.info(f"Event path {event_path} not found")
                return None, None
                
            # Extract timestamps
            ts_path = f"{event_path}/timestamps"
            if ts_path in h5_file:
                ts_dataset = h5_file[ts_path]
                self.logger.info(f"Found {event_type} timestamps: shape {ts_dataset.shape}")
                timestamps = ts_dataset[:].tolist()
            else:
                self.logger.warning(f"No timestamps found at {ts_path}")
                
            # Extract event data
            data_path = f"{event_path}/data"
            if data_path in h5_file:
                data_dataset = h5_file[data_path]
                self.logger.info(f"Found {event_type} data: shape {data_dataset.shape}")
                
                # Process data items - handle byte strings, convert to Python objects
                processed_data = []
                for item in data_dataset[:]:
                    if isinstance(item, bytes):
                        try:
                            # Try to decode as UTF-8 string
                            decoded = item.decode('utf-8')
                            # If it looks like JSON, parse it
                            if decoded.startswith('{') and decoded.endswith('}'):
                                try:
                                    import json
                                    processed_data.append(json.loads(decoded))
                                except json.JSONDecodeError:
                                    processed_data.append(decoded)
                            else:
                                processed_data.append(decoded)
                        except UnicodeDecodeError:
                            processed_data.append(str(item))
                    else:
                        processed_data.append(item)
                
                data = processed_data
            else:
                self.logger.warning(f"No data found at {data_path}")
                
            return timestamps, data
            
        except Exception as e:
            self.logger.error(f"Error extracting {event_type} data: {e}")
            return None, None

    def process_session(self, session: Session) -> Dict:
        """Process a single session to log H5 structure for audio extraction.

        Args:
            session: Session object

        Returns:
            Dict with processing results
        """
        session_id = session.session_id
        self.logger.info(f"Processing session for audio H5 structure logging: {session_id}")

        curated_h5_key = f"{self.source_prefix}{session_id}.h5"

        # Check file existence first (optional but good practice)
        try:
            # Use the s3 client provided by BaseTransform
            self.s3.head_object(Bucket=self.s3_bucket, Key=curated_h5_key)
            self.logger.info(f"Confirmed H5 file exists: {curated_h5_key}")
        except Exception as e:
            # Use self.s3.exceptions.ClientError if using boto3 explicitly and need specific error handling
            self.logger.error(f"H5 file not found or inaccessible for session {session_id} at {curated_h5_key}: {e}")
            return {
                "status": "failed",
                "error_details": f"H5 file not found or inaccessible: {curated_h5_key}",
                "metadata": {"session_id": session_id},
                "files_to_copy": [],
                "files_to_upload": []
            }

        # Download the file using the Session object
        local_h5_path = session.download_file(curated_h5_key)
        if not local_h5_path:
             # download_file should ideally raise an error or return None/False on failure
             self.logger.error(f"Failed to download H5 file for session {session_id} from {curated_h5_key}")
             return {
                "status": "failed",
                "error_details": f"Failed to download H5 file: {curated_h5_key}",
                "metadata": {"session_id": session_id},
                "files_to_copy": [],
                "files_to_upload": []
            }

        logged_info = {"session_id": session_id, "file_key": curated_h5_key, "datasets": {}, "groups": {}}
        audio_data = None
        all_audio_timestamps = None
        samples_per_chunk = 4096  # Default value
        
        try:
            # Open the downloaded H5 file
            with h5py.File(local_h5_path, 'r') as hf:
                self.logger.info(f"Successfully opened H5 file: {local_h5_path}")

                # Log audio group attributes
                audio_group_attrs = self._log_group_attributes(hf, "/audio")
                if audio_group_attrs is not None:
                    logged_info["groups"]["/audio"] = audio_group_attrs

                # Log audio data dataset info
                audio_data_info = self._log_dataset_info(hf, "/audio/audio_data")
                if audio_data_info:
                    logged_info["datasets"]["/audio/audio_data"] = audio_data_info
                    # Get the samples_per_chunk from the shape for later use
                    if 'shape' in audio_data_info and len(audio_data_info['shape']) > 1:
                        samples_per_chunk = audio_data_info['shape'][1]
                    # Check for sampling rate in attributes
                    if 'attributes' in audio_data_info:
                        for sr_key in ['sampling_rate', 'sample_rate', 'fs', 'samplerate']:
                             if sr_key in audio_data_info['attributes']:
                                self.logger.info(f"Potential sampling rate found in /audio/audio_data attributes: {sr_key} = {audio_data_info['attributes'][sr_key]}")
                    
                    # Extract the actual audio data while file is open
                    try:
                        self.logger.info(f"Loading audio data from {audio_data_info['shape']}...")
                        audio_data = hf["/audio/audio_data"][:]
                        self.logger.info(f"Successfully loaded audio data: {audio_data.shape}")
                    except Exception as e:
                        self.logger.error(f"Error loading audio data: {e}")
                        # Continue even if we can't load audio data - we'll still generate the report

                # Log audio timestamps dataset info
                audio_ts_info = self._log_dataset_info(hf, "/audio/timestamps")
                if audio_ts_info:
                    logged_info["datasets"]["/audio/timestamps"] = audio_ts_info
                    # Extract first 10 rows of audio timestamps
                    audio_ts_data = self._extract_timestamps(hf, "/audio/timestamps", limit=10)
                    if audio_ts_data:
                        self.logger.info(f"Extracted first 10 rows of audio timestamps")
                        logged_info["data"] = logged_info.get("data", {})
                        logged_info["data"]["audio_timestamps_first_10"] = audio_ts_data
                    
                    # Extract all timestamps while file is open
                    try:
                        all_audio_timestamps = hf["/audio/timestamps"][:]
                        self.logger.info(f"Successfully loaded all audio timestamps: {all_audio_timestamps.shape}")
                    except Exception as e:
                        self.logger.error(f"Error loading all audio timestamps: {e}")
                        # Continue without full timestamps

                # Log audio chunk metadata dataset info (if exists)
                audio_chunk_meta_info = self._log_dataset_info(hf, "/audio/chunk_metadata")
                if audio_chunk_meta_info:
                    logged_info["datasets"]["/audio/chunk_metadata"] = audio_chunk_meta_info
                
                # Log recording start event timestamps dataset info
                rec_start_info = self._log_dataset_info(hf, "/events/recording_start/timestamps")
                if rec_start_info:
                    logged_info["datasets"]["/events/recording_start/timestamps"] = rec_start_info
                    # Extract all recording start timestamps
                    rec_start_data = self._extract_timestamps(hf, "/events/recording_start/timestamps")
                    if rec_start_data:
                        self.logger.info(f"Extracted all recording start timestamps: {len(rec_start_data)} rows")
                        logged_info["data"] = logged_info.get("data", {})
                        logged_info["data"]["recording_start_timestamps"] = rec_start_data

                # Log recording stop event timestamps dataset info
                rec_stop_info = self._log_dataset_info(hf, "/events/recording_stop/timestamps")
                if rec_stop_info:
                    logged_info["datasets"]["/events/recording_stop/timestamps"] = rec_stop_info
                    # Extract all recording stop timestamps
                    rec_stop_data = self._extract_timestamps(hf, "/events/recording_stop/timestamps")
                    if rec_stop_data:
                        self.logger.info(f"Extracted all recording stop timestamps: {len(rec_stop_data)} rows")
                        logged_info["data"] = logged_info.get("data", {})
                        logged_info["data"]["recording_stop_timestamps"] = rec_stop_data
                
                # Log and extract speech transcription data if available
                self.logger.info("Looking for speech transcription data in /language/S/words")
                speech_items = self._extract_language_speech(hf)
                if speech_items:
                    # Add speech data to logged_info
                    logged_info["data"] = logged_info.get("data", {})
                    logged_info["data"]["speech_transcription"] = speech_items
                    # Log the first few words as a sample
                    sample_size = min(5, len(speech_items))
                    sample_words = [item.get('word', '[unknown]') for item in speech_items[:sample_size]]
                    self.logger.info(f"Speech transcription sample: {', '.join(sample_words)}...")
                    
                    # Log group attributes for the speech group
                    speech_group_attrs = self._log_group_attributes(hf, "/language/S")
                    if speech_group_attrs:
                        logged_info["groups"]["/language/S"] = speech_group_attrs
                        
                    # Add the dataset info to logged_info
                    if "/language/S/words" in hf:
                        speech_dataset_info = self._log_dataset_info(hf, "/language/S/words")
                        if speech_dataset_info:
                            logged_info["datasets"]["/language/S/words"] = speech_dataset_info
                
                # Extract element events (element_sent and element_replied)
                self.logger.info("Extracting element events (element_sent and element_replied)")
                
                # Extract element_sent events
                element_sent_ts, element_sent_data = self._extract_element_events(hf, "element_sent")
                if element_sent_ts:
                    logged_info["data"]["element_sent_timestamps"] = element_sent_ts
                    if element_sent_data:
                        logged_info["data"]["element_sent_data"] = element_sent_data
                    self.logger.info(f"Extracted {len(element_sent_ts)} element_sent events")
                
                # Extract element_replied events
                element_replied_ts, element_replied_data = self._extract_element_events(hf, "element_replied")
                if element_replied_ts:
                    logged_info["data"]["element_replied_timestamps"] = element_replied_ts
                    if element_replied_data:
                        logged_info["data"]["element_replied_data"] = element_replied_data
                    self.logger.info(f"Extracted {len(element_replied_ts)} element_replied events")
                
                # Log dataset info for element events
                for event_type in ["element_sent", "element_replied"]:
                    event_path = f"/events/{event_type}"
                    if event_path in hf:
                        # Log group attributes
                        event_attrs = self._log_group_attributes(hf, event_path)
                        if event_attrs:
                            logged_info["groups"][event_path] = event_attrs
                        
                        # Log timestamps dataset info
                        ts_path = f"{event_path}/timestamps"
                        if ts_path in hf:
                            ts_info = self._log_dataset_info(hf, ts_path)
                            if ts_info:
                                logged_info["datasets"][ts_path] = ts_info
                        
                        # Log data dataset info
                        data_path = f"{event_path}/data"
                        if data_path in hf:
                            data_info = self._log_dataset_info(hf, data_path)
                            if data_info:
                                logged_info["datasets"][data_path] = data_info

            # Define metadata to be potentially saved to DynamoDB
            metadata = {
                "session_id": session_id,
                "audio_data_found": "/audio/audio_data" in logged_info["datasets"],
                "audio_timestamps_found": "/audio/timestamps" in logged_info["datasets"],
                "rec_start_events_found": "/events/recording_start/timestamps" in logged_info["datasets"],
                "rec_stop_events_found": "/events/recording_stop/timestamps" in logged_info["datasets"],
                "samples_per_chunk": samples_per_chunk
            }

            # Now extract and save audio segments between recording start/stop events
            # We'll add the report to this list after processing
            files_to_upload = []
            
            # Process audio data if we have it
            if audio_data is not None:
                # Convert all_audio_timestamps to the format needed by _extract_audio_segments
                # if we have the raw timestamps from h5py
                if all_audio_timestamps is not None and not isinstance(all_audio_timestamps, list):
                    all_audio_timestamps = all_audio_timestamps.tolist()
                else:
                    # Use the first 10 timestamps we extracted earlier if that's all we have
                    all_audio_timestamps = logged_info["data"].get("audio_timestamps_first_10", [])
                
                # Calculate sample rate from timestamps
                sample_rate = self._estimate_sample_rate(
                    all_audio_timestamps[:20] if len(all_audio_timestamps) >= 20 else all_audio_timestamps, 
                    samples_per_chunk
                )
                
                # Calculate statistics for entire audio stream using 10-second windows
                self.logger.info("Calculating audio statistics for 10-second windows...")
                audio_stats = self._calculate_audio_stats(audio_data, sample_rate, window_sec=10.0)
                
                # Identify outlier windows
                self.logger.info("Identifying outlier windows...")
                outliers = self._identify_outliers(audio_stats, z_threshold=3.0)
                
                # Add stats to logged_info
                if "statistics" not in logged_info:
                    logged_info["statistics"] = {}
                
                logged_info["statistics"]["global_stats"] = audio_stats["global"]
                logged_info["statistics"]["outlier_count"] = len(outliers)
                logged_info["statistics"]["outliers"] = outliers[:10]  # Only include first 10 outliers to keep size reasonable
                self.logger.info(f"Found {len(outliers)} outlier windows in audio data")
                self.logger.info(f"Added statistics to JSON report - keys: {list(logged_info.keys())}")
                
                # Only proceed with extraction if we have the necessary timestamps
                if "recording_start_timestamps" in logged_info.get("data", {}) and "recording_stop_timestamps" in logged_info.get("data", {}):
                    # Extract segments
                    segments = self._extract_audio_segments(
                        audio_data,
                        all_audio_timestamps,
                        logged_info["data"]["recording_start_timestamps"],
                        logged_info["data"]["recording_stop_timestamps"],
                        sample_rate
                    )
                    
                    # Calculate statistics for the extracted segments
                    segment_stats = self._calculate_segment_stats(segments, sample_rate)
                    if "statistics" not in logged_info:
                        logged_info["statistics"] = {}
                    logged_info["statistics"]["segments"] = segment_stats
                    self.logger.info(f"Added segment statistics to JSON report: {len(segment_stats)} segments")
                
                    # Save each segment as a WAV file
                    for segment_data, segment_name in segments:
                        try:
                            local_wav_path = self._save_wav_file(session, segment_data, sample_rate, segment_name)
                            wav_dest_key = f"{self.destination_prefix}{os.path.basename(local_wav_path)}"
                            files_to_upload.append((local_wav_path, wav_dest_key))
                        except Exception as e:
                            self.logger.error(f"Error saving WAV file for segment {segment_name}: {e}")
                            # Continue with other segments
                
                    # Update metadata with segment info
                    metadata["segments_extracted"] = len(segments)
                    metadata["sample_rate"] = sample_rate
                    metadata["min_segment_duration"] = self.min_segment_duration
                    metadata["normalize_audio"] = self.normalize_audio
                    metadata["outlier_count"] = len(outliers)
                    
                    # Add speech transcription metadata if available
                    if "speech_transcription" in logged_info.get("data", {}):
                        metadata["speech_items_count"] = len(logged_info["data"]["speech_transcription"])
                    
                    # Add element events metadata if available
                    if "element_sent_timestamps" in logged_info.get("data", {}):
                        metadata["element_sent_count"] = len(logged_info["data"]["element_sent_timestamps"])
                    if "element_replied_timestamps" in logged_info.get("data", {}):
                        metadata["element_replied_count"] = len(logged_info["data"]["element_replied_timestamps"])
                    
                    if len(segments) == 0:
                        self.logger.warning("No audio segments were extracted. Check recording events and min_segment_duration.")
                else:
                    self.logger.warning("Missing recording start/stop events, skipping audio extraction")
                    metadata["segments_extracted"] = 0
                    metadata["extraction_skipped"] = True
            else:
                self.logger.warning("No audio data could be loaded, skipping audio extraction")
                metadata["segments_extracted"] = 0
                metadata["extraction_skipped"] = True
        
            # Save the logged info as a JSON report after all processing is done
            report_file_name = f"{session_id}_audio_struct_report.json"
            local_report_path = session.create_upload_file(report_file_name)
                
            # Log keys in logged_info before saving
            self.logger.info(f"JSON report keys before saving: {list(logged_info.keys())}")
            if "statistics" in logged_info:
                self.logger.info(f"Statistics keys: {list(logged_info['statistics'].keys())}")
                
            with open(local_report_path, 'w') as f:
                json.dump(logged_info, f, indent=2)
                
            self.logger.info(f"Saved JSON report to {local_report_path}")

            # Define the destination key for the report
            dest_key = f"{self.destination_prefix}{report_file_name}"
                
            # Add report to the files to upload
            files_to_upload.append((local_report_path, dest_key))
                
            # Return success status and files to upload
            return {
                "status": "success",
                "metadata": metadata,
                "files_to_copy": [],
                "files_to_upload": files_to_upload
            }

        except Exception as e:
            self.logger.error(f"Error processing H5 file {local_h5_path} for session {session_id}: {e}", exc_info=True)
            # Return failure status
            return {
                "status": "failed",
                "error_details": str(e),
                "metadata": {"session_id": session_id},
                "files_to_copy": [],
                "files_to_upload": []
            }
        # No finally block needed here for cleanup, BaseTransform handles it via Session context manager

    @classmethod
    def add_subclass_arguments(cls, parser):
        """Add audio-extraction-specific command-line arguments.

        Args:
            parser: ArgumentParser instance
        """
        parser.add_argument('--min-segment-duration', type=float, default=0.5,
                          help='Minimum duration in seconds for extracted audio segments (default: 0.5)')
        parser.add_argument('--normalize-audio', action='store_true', default=True,
                          help='Normalize audio data to range [-1.0, 1.0] (default: True)')
        parser.add_argument('--force-sample-rate', type=int, default=None,
                          help='Force specific sample rate instead of auto-detecting (e.g., 16000, 44100)')
        parser.add_argument('--stats-window-size', type=float, default=10.0,
                          help='Window size in seconds for audio statistics (default: 10.0)')
        parser.add_argument('--outlier-threshold', type=float, default=3.0,
                          help='Z-score threshold for identifying outlier windows (default: 3.0)')

    @classmethod
    def from_args(cls, args):
        """Create a transform instance from command-line arguments.

        Args:
            args: Parsed command-line arguments

        Returns:
            Instance of AudioExtractTransform
        """
        # Ensure all necessary args from BaseTransform.add_arguments are passed
        # Use getattr for safety, providing defaults where appropriate
        return cls(
            source_prefix=getattr(args, 'source_prefix', cls.SOURCE_PREFIX),
            destination_prefix=getattr(args, 'dest_prefix', cls.DEST_PREFIX),
            s3_bucket=getattr(args, 's3_bucket', None), # BaseTransform should handle missing bucket
            verbose=getattr(args, 'verbose', False),
            log_file=getattr(args, 'log_file', None),
            dry_run=getattr(args, 'dry_run', False),
            keep_local=getattr(args, 'keep_local', False),
            # Add specific args defined in add_subclass_arguments:
            min_segment_duration=getattr(args, 'min_segment_duration', 0.5),
            normalize_audio=getattr(args, 'normalize_audio', True),
            force_sample_rate=getattr(args, 'force_sample_rate', None),
            stats_window_size=getattr(args, 'stats_window_size', 10.0),
            outlier_threshold=getattr(args, 'outlier_threshold', 3.0),
        )

# Entry point for running the transform directly from the command line (optional)
# This part is generally not used when transforms are invoked via the main pipeline CLI (cli.py)
if __name__ == "__main__":
    # This allows running `python t2F_audio_v0.py --session-id XYZ ...` for testing
    # It relies on BaseTransform's run_from_command_line class method
    AudioExtractTransform.run_from_command_line()