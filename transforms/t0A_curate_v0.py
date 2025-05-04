"""
Data curation transform for processing raw sessions.

This transform:
1. Calculates metrics for sessions from data-collector/
2. Filters sessions based on clean data time and data rate
3. Copies qualifying main H5 files to curated-h5/
4. Records metadata in DynamoDB

This is implemented using the new BaseTransform architecture.
"""

import os
import sys
import json
import argparse
from typing import Dict, Any, List, Optional, Set, Tuple

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import base transform and utilities
from base_transform import BaseTransform, Session
from utils.coarse_data_metrics import (
    classify_data_rate, calculate_data_rate,
    find_task_events_file, find_element_events_file, 
    get_client_timestamps_from_events
)


class CurateTransform(BaseTransform):
    """Data curation transform for processing raw sessions.

    This transform identifies sessions with sufficient data quality and
    duration, and copies them to a curated directory for further processing.
    """
    
    # Define required class attributes for source and destination prefixes
    SOURCE_PREFIX = 'data-collector/new-sessions/'
    DEST_PREFIX = 'curated-h5/'

    def __init__(self, min_clean_duration_sec: float = 30.0,
                 min_data_rate_classification: str = "fnirs only",
                 **kwargs):
        """Initialize the curation transform.

        Args:
            min_clean_duration_sec: Minimum clean duration in seconds
            min_data_rate_classification: Minimum data rate classification
            **kwargs: Additional arguments for BaseTransform
        """
        # Set default transform info if not provided
        transform_id = kwargs.pop('transform_id', 'curation_v1')
        script_id = kwargs.pop('script_id', '0A')
        script_name = kwargs.pop('script_name', 'curate_sessions')
        script_version = kwargs.pop('script_version', 'v0')

        # Call parent constructor
        super().__init__(
            transform_id=transform_id,
            script_id=script_id,
            script_name=script_name,
            script_version=script_version,
            **kwargs
        )

        # Set curation-specific attributes
        self.min_clean_duration_sec = min_clean_duration_sec
        self.min_data_rate_classification = min_data_rate_classification

        self.acceptable_classifications = ["fnirs only", "both (fnirs+eeg)"]
        if self.min_data_rate_classification == "both (fnirs+eeg)":
            # If we require both, remove fnirs only from acceptable
            self.acceptable_classifications = ["both (fnirs+eeg)"]

        self.logger.info(f"Curation parameters:")
        self.logger.info(f"  Min clean duration: {self.min_clean_duration_sec} seconds")
        self.logger.info(f"  Min data rate classification: {self.min_data_rate_classification}")

    def process_session(self, session: Session) -> Dict:
        """Process a single session.
        
        This implementation:
        1. Calculates session metrics (data rate, duration)
        2. Determines if the session qualifies for curation
        3. Returns copy operations for qualifying sessions
        
        Args:
            session: Session object
            
        Returns:
            Dict with processing results
        """
        session_id = session.session_id
        self.logger.info(f"Processing session: {session_id}")
        
        # Calculate session metrics
        metrics = self.calculate_session_metrics(session)
        
        # If session doesn't qualify, return early
        if not metrics["qualifies"]:
            self.logger.info(f"Session {session_id} does NOT qualify for curation")
            
            # List reasons why it didn't qualify
            reasons = []
            if metrics["clean_duration_sec"] < self.min_clean_duration_sec:
                reasons.append(f"Clean duration ({metrics['clean_duration_sec']:.2f}s) < minimum ({self.min_clean_duration_sec}s)")
            if metrics["classification"] not in self.acceptable_classifications:
                reasons.append(f"Classification '{metrics['classification']}' doesn't meet minimum '{self.min_data_rate_classification}'")
            self.logger.debug(f"Reason(s): {', '.join(reasons)}")
            
            return {
                "metadata": metrics,
                "status": "skipped",
                "files_to_copy": [],
                "files_to_upload": []
            }
            
        # Session qualifies
        self.logger.info(f"Session {session_id} QUALIFIES for curation")
        
        # Prepare copy operations for all H5 files
        files_to_copy = []
        for file_info in metrics["main_h5_files"]:
            source_key = file_info["key"]
            file_name = os.path.basename(source_key)
            dest_key = f"{self.destination_prefix}{file_name}"
            
            # Add copy operation
            files_to_copy.append(session.copy(source_key, dest_key))
            
        return {
            "metadata": metrics,
            "status": "success",
            "files_to_copy": files_to_copy,
            "files_to_upload": []
        }

    def calculate_session_metrics(self, session: Session) -> Dict:
        """Calculate metrics for a session.
        
        Args:
            session: Session object
            
        Returns:
            Dict of session metrics
        """
        self.logger.debug(f"Calculating metrics for session {session.session_id}")
        
        # Initialize session metrics
        metrics = {
            "id": session.session_id,
            "clean_duration_sec": 0,
            "file_duration_sec": 0,
            "data_rate": 0,
            "classification": "neither",
            "main_h5_files": [],
            "total_size_bytes": 0,
            "qualifies": False
        }
        
        # Build the session path
        session_prefix = f"{self.source_prefix}{session.session_id}/"
        
        # Log exactly what files we find
        all_files = session.list_files() 
        self.logger.debug(f"All files in session: {all_files}")
        
        # Check for element_events.json file
        element_events_files = [key for key in all_files if key.endswith('element_events.json')]
        element_events_key = element_events_files[0] if element_events_files else None
        
        if element_events_key:
            self.logger.debug(f"Found element_events.json: {element_events_key}")
            
            # Get timestamps from element events
            first_timestamp, last_timestamp = get_client_timestamps_from_events(self.s3, element_events_key)
            if first_timestamp is None or last_timestamp is None:
                self.logger.info(f"Could not extract timestamps from element_events, duration is 0")
                clean_duration_sec = 0
                file_duration_sec = 0
            else:
                # Calculate duration
                clean_duration_ms = last_timestamp - first_timestamp
                clean_duration_sec = clean_duration_ms / 1000
                file_duration_sec = clean_duration_sec  # Same duration for now
                self.logger.debug(f"Duration from element events: {clean_duration_sec:.2f} seconds")
                metrics["clean_duration_sec"] = clean_duration_sec
                metrics["file_duration_sec"] = file_duration_sec
        else:
            self.logger.info(f"No element_events.json found, duration is 0")
            clean_duration_sec = 0
            file_duration_sec = 0
        
        # Get all H5 files and their sizes
        total_size = 0
        h5_files = []
        
        # Get H5 files in the session
        h5_file_keys = session.list_files(extension='.h5')
        self.logger.debug(f"Found {len(h5_file_keys)} H5 files: {h5_file_keys}")
        
        # Get sizes of the H5 files
        file_sizes = session.get_file_sizes(extension='.h5')
        
        # Process each H5 file
        for file_key in h5_file_keys:
            file_size = file_sizes.get(file_key, 0)
            h5_files.append({"key": file_key, "size": file_size})
            total_size += file_size
        
        metrics["main_h5_files"] = h5_files
        metrics["total_size_bytes"] = total_size
        
        self.logger.debug(f"Found {len(h5_files)} main H5 files")
        self.logger.debug(f"Total data size: {total_size/(1024*1024):.2f} MB")
        
        # If we have no H5 files, return early
        if not h5_files:
            self.logger.warning(f"No H5 files found in session {session.session_id}")
            return metrics
        
        # Calculate data rate using file duration
        if file_duration_sec and file_duration_sec > 0:
            # Convert to milliseconds for the calculation function
            file_duration_ms = file_duration_sec * 1000
            data_rate = calculate_data_rate(total_size, file_duration_ms)
            classification = classify_data_rate(data_rate)
            metrics["data_rate"] = data_rate
            metrics["classification"] = classification
            self.logger.debug(f"Data rate: {data_rate:.2f} KB/s - Classification: {classification}")
        
        # Determine if session qualifies for curation
        if (metrics["clean_duration_sec"] >= self.min_clean_duration_sec and
                metrics["classification"] in self.acceptable_classifications):
            metrics["qualifies"] = True
        
        return metrics

    @classmethod
    def add_subclass_arguments(cls, parser):
        """Add curation-specific command-line arguments.

        Args:
            parser: ArgumentParser instance
        """
        parser.add_argument('--min-clean-duration', type=float, default=30.0,
                          help='Minimum clean duration in seconds')
        parser.add_argument('--min-data-rate', type=str, default='fnirs only',
                          choices=['fnirs only', 'both (fnirs+eeg)'],
                          help='Minimum data rate classification')

    @classmethod
    def from_args(cls, args):
        """Create a transform instance from command-line arguments.

        Args:
            args: Parsed command-line arguments

        Returns:
            Instance of CurateTransform
        """
        # Extract arguments - using defaults from class if not provided
        source_prefix = getattr(args, 'source_prefix', cls.SOURCE_PREFIX)
        dest_prefix = getattr(args, 'dest_prefix', cls.DEST_PREFIX)
        
        return cls(
            source_prefix=source_prefix,
            destination_prefix=dest_prefix,
            min_clean_duration_sec=args.min_clean_duration,
            min_data_rate_classification=args.min_data_rate,
            s3_bucket=args.s3_bucket,
            verbose=args.verbose,
            log_file=args.log_file,
            dry_run=args.dry_run,  # Passing to base class only
            keep_local=args.keep_local  # Passing to base class only
        )


# Entry point for running the transform from the command line
if __name__ == "__main__":
    CurateTransform.run_from_command_line()