"""
Data curation transform for processing raw sessions.

This transform:
1. Calculates metrics for sessions from data-collector/
2. Filters sessions based on clean data time and data rate
3. Copies qualifying main H5 files to curated-h5/
4. Records metadata in DynamoDB
"""

import os
import sys
import boto3
from typing import Dict, Any, List, Optional, Set

try:
    # When running as an installed package
    from ..base import DataTransform
    from ..utils.coarse_data_metrics import (
        classify_data_rate, calculate_data_rate,
        find_task_events_file, find_element_events_file, 
        get_client_timestamps_from_events, get_session_files_and_size
    )
except ImportError:
    # When running as a script
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from base import DataTransform
    from utils.coarse_data_metrics import (
        classify_data_rate, calculate_data_rate,
        find_task_events_file, find_element_events_file,
        get_client_timestamps_from_events, get_session_files_and_size
    )


class CurationTransform(DataTransform):
    """Data curation transform for processing raw sessions.

    This transform identifies sessions with sufficient data quality and
    duration, and copies them to a curated directory for further processing.
    """

    def __init__(self, source_prefix: str = 'data-collector/new-sessions/',
                dest_prefix: str = 'curated-h5/',
                min_clean_duration_sec: float = 30.0,
                min_data_rate_classification: str = "fnirs only",
                **kwargs):
        """Initialize the curation transform.

        Args:
            source_prefix: S3 prefix for source data
            dest_prefix: S3 prefix for destination data
            min_clean_duration_sec: Minimum clean duration in seconds
            min_data_rate_classification: Minimum data rate classification
            **kwargs: Additional arguments for DataTransform
        """
        # Set default transform info if not provided
        transform_id = kwargs.pop('transform_id', 'curation_v0')
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
        self.source_prefix = source_prefix
        self.dest_prefix = dest_prefix
        self.min_clean_duration_sec = min_clean_duration_sec
        self.min_data_rate_classification = min_data_rate_classification

        self.acceptable_classifications = ["fnirs only", "both (fnirs+eeg)"]
        if self.min_data_rate_classification == "both (fnirs+eeg)":
            # If we require both, remove fnirs only from acceptable
            self.acceptable_classifications = ["both (fnirs+eeg)"]

        self.logger.info(f"Curation transform initialized with:")
        self.logger.info(f"  Source prefix: {self.source_prefix}")
        self.logger.info(f"  Destination prefix: {self.dest_prefix}")
        self.logger.info(f"  Min clean duration: {self.min_clean_duration_sec} seconds")
        self.logger.info(f"  Min data rate classification: {self.min_data_rate_classification}")

    def find_items_to_process(self):
        """Find sessions that need to be curated.

        Returns:
            List of session IDs to process
        """
        self.logger.info(f"Listing sessions in {self.source_prefix}")
        paginator = self.s3.get_paginator('list_objects_v2')
        sessions = []

        for page in paginator.paginate(Bucket=self.s3_bucket, Prefix=self.source_prefix, Delimiter='/'):
            if 'CommonPrefixes' in page:
                for prefix_obj in page['CommonPrefixes']:
                    session_prefix = prefix_obj['Prefix']
                    session_id = session_prefix.strip('/').split('/')[-1]
                    sessions.append(session_id)

        self.logger.info(f"Found {len(sessions)} sessions")
        return sessions

    def calculate_session_metrics(self, session_id):
        """Calculate metrics for a session.

        Args:
            session_id: Session ID to process

        Returns:
            Dict of session metrics
        """
        # If we're getting a full H5 file path, extract the session ID
        original_session_id = session_id
        
        # Check if this is a full H5 file path - handle both files in root and in subfolders
        if session_id.startswith(self.source_prefix) and session_id.endswith('.h5'):
            # Remove the source prefix
            path_without_prefix = session_id[len(self.source_prefix):]
            
            # Extract the session ID (first part of the path)
            session_id = path_without_prefix.split('/')[0]
            self.logger.info(f"Extracted session ID '{session_id}' from H5 file path '{original_session_id}'")
            
        # Set the proper session prefix, always at the top level
        session_prefix = f"{self.source_prefix}{session_id}/"
        self.logger.debug(f"Calculating metrics for session {session_id}")

        # Initialize session metrics
        metrics = {
            "id": session_id,
            "clean_duration_sec": 0,
            "file_duration_sec": 0,
            "data_rate": 0,
            "classification": "neither",
            "main_h5_files": [],
            "total_size_bytes": 0,
            "qualifies": False
        }

        # Check for element_events.json
        element_events_key = find_element_events_file(self.s3, session_prefix)
        if not element_events_key:
            self.logger.info(f"No element_events.json found in session {session_id}, trying task_events.json as fallback")
            
            # Fallback to task_events.json
            task_events_key = find_task_events_file(self.s3, session_prefix)
            if not task_events_key:
                self.logger.info(f"No element_events.json or task_events.json found in session {session_id}, duration is 0")
                clean_duration_ms = 0
                clean_duration_sec = 0
                file_duration_ms = 0
                file_duration_sec = 0
            else:
                self.logger.debug(f"Found task_events.json: {task_events_key}")
                
                # Get timestamps from element events (fallback to task events)
                first_timestamp, last_timestamp = get_client_timestamps_from_events(self.s3, task_events_key)
                if first_timestamp is None or last_timestamp is None:
                    self.logger.info(f"Could not extract timestamps from element events for {session_id}, duration is 0")
                    clean_duration_ms = 0
                    clean_duration_sec = 0
                    file_duration_ms = 0
                    file_duration_sec = 0
                else:
                    # Use the same timestamps for both clean and file duration
                    clean_duration_ms = last_timestamp - first_timestamp
                    clean_duration_sec = clean_duration_ms / 1000
                    file_duration_ms = clean_duration_ms  # Same duration for now
                    file_duration_sec = clean_duration_sec  # Same duration for now
                    self.logger.debug(f"Duration from element events (task events fallback): {clean_duration_sec:.2f} seconds")
                    metrics["clean_duration_sec"] = clean_duration_sec
                    metrics["file_duration_sec"] = file_duration_sec
        else:
            self.logger.debug(f"Found element_events.json: {element_events_key}")
            
            # Get timestamps from element events
            first_timestamp, last_timestamp = get_client_timestamps_from_events(self.s3, element_events_key)
            if first_timestamp is None or last_timestamp is None:
                self.logger.info(f"Could not extract timestamps from element_events for {session_id}, duration is 0")
                clean_duration_ms = 0
                clean_duration_sec = 0
                file_duration_ms = 0
                file_duration_sec = 0
            else:
                # Use the same timestamps for both clean and file duration
                clean_duration_ms = last_timestamp - first_timestamp
                clean_duration_sec = clean_duration_ms / 1000
                file_duration_ms = clean_duration_ms  # Same duration for now
                file_duration_sec = clean_duration_sec  # Same duration for now
                self.logger.debug(f"Duration from element events: {clean_duration_sec:.2f} seconds")
                metrics["clean_duration_sec"] = clean_duration_sec
                metrics["file_duration_sec"] = file_duration_sec

        # Get all session files
        main_h5_files, sub_h5_files, total_size = get_session_files_and_size(self.s3, session_prefix)
        metrics["total_size_bytes"] = total_size

        # Get file paths for main H5 files (needed for copying later)
        for file_key, file_size in main_h5_files:
            metrics["main_h5_files"].append({"key": file_key, "size": file_size})

        if not main_h5_files and not sub_h5_files:
            self.logger.warning(f"No H5 files found in session {session_id}")
            return metrics

        self.logger.debug(f"Total data size: {total_size/(1024*1024):.2f} MB")

        # Calculate data rate using file duration
        if file_duration_ms and file_duration_ms > 0:
            data_rate = calculate_data_rate(total_size, file_duration_ms)
            classification = classify_data_rate(data_rate)
            metrics["data_rate"] = data_rate
            metrics["classification"] = classification
            self.logger.debug(f"Data rate (using file duration): {data_rate:.2f} KB/s - Classification: {classification}")

        # Determine if session qualifies for curation
        if (metrics["clean_duration_sec"] >= self.min_clean_duration_sec and
                metrics["classification"] in self.acceptable_classifications):
            metrics["qualifies"] = True
            self.logger.info(f"Session {session_id} QUALIFIES for curation")
        else:
            self.logger.info(f"Session {session_id} does NOT qualify for curation")

            # List reasons why it didn't qualify
            reasons = []
            if metrics["clean_duration_sec"] < self.min_clean_duration_sec:
                reasons.append(f"Clean duration ({metrics['clean_duration_sec']:.2f}s) < minimum ({self.min_clean_duration_sec}s)")
            if metrics["classification"] not in self.acceptable_classifications:
                reasons.append(f"Classification '{metrics['classification']}' doesn't meet minimum '{self.min_data_rate_classification}'")
            self.logger.debug(f"Reason(s): {', '.join(reasons)}")

        return metrics

    def copy_session_to_curated(self, session_metrics):
        """Copy qualifying session data to curated folder.

        Args:
            session_metrics: Metrics for the session

        Returns:
            Dict with transform record or None if session doesn't qualify
        """
        session_id = session_metrics["id"]

        # Check if session qualifies - don't record skipped sessions
        if not session_metrics["qualifies"]:
            self.logger.debug(f"Session {session_id} doesn't qualify for curation, skipping")
            return None

        if not session_metrics["main_h5_files"]:
            self.logger.warning(f"Session {session_id} has no main H5 files to copy")

            # Create a record for session with no files
            error_details = 'No main H5 files found for session'
            return self.record_transform(
                data_id=session_id,
                transform_metadata={
                    'clean_duration_sec': session_metrics["clean_duration_sec"],
                    'file_duration_sec': session_metrics["file_duration_sec"],
                    'data_rate': session_metrics["data_rate"],
                    'classification': session_metrics["classification"],
                    'total_size_bytes': session_metrics["total_size_bytes"]
                },
                status='failed',
                error_details=error_details
            )

        # For each main H5 file, copy to curated folder
        copied_files = []
        source_paths = []
        try:
            for h5_file in session_metrics["main_h5_files"]:
                source_key = h5_file["key"]
                file_name = source_key.split('/')[-1]

                # Create destination key directly in destination folder without subdirectory
                dest_key = f"{self.dest_prefix}{file_name}"

                # Add to copied files even in dry run for complete record preview
                source_path = f"s3://{self.s3_bucket}/{source_key}"
                dest_path = f"s3://{self.s3_bucket}/{dest_key}"

                source_paths.append(source_path)
                copied_files.append({
                    "source": source_path,
                    "destination": dest_path,
                    "size_bytes": h5_file["size"]
                })

                if self.dry_run:
                    self.logger.info(f"[DRY RUN] Would copy {source_path} to {dest_path}")
                else:
                    self.logger.info(f"Copying {source_path} to {dest_path}")
                    # Use copy_object for efficiency (stays server-side in S3)
                    self.s3.copy_object(
                        Bucket=self.s3_bucket,
                        CopySource={'Bucket': self.s3_bucket, 'Key': source_key},
                        Key=dest_key
                    )

            # Record successful transform
            return self.record_transform(
                data_id=session_id,
                transform_metadata={
                    'clean_duration_sec': session_metrics["clean_duration_sec"],
                    'file_duration_sec': session_metrics["file_duration_sec"],
                    'data_rate': session_metrics["data_rate"],
                    'classification': session_metrics["classification"],
                    'total_size_bytes': session_metrics["total_size_bytes"]
                },
                source_paths=source_paths,
                destination_paths=[item["destination"] for item in copied_files],
                status='success'
            )

        except Exception as e:
            self.logger.error(f"Error copying files for session {session_id}: {e}")

            # Record failed transform
            return self.record_transform(
                data_id=session_id,
                transform_metadata={
                    'clean_duration_sec': session_metrics["clean_duration_sec"],
                    'file_duration_sec': session_metrics["file_duration_sec"],
                    'data_rate': session_metrics["data_rate"],
                    'classification': session_metrics["classification"],
                    'total_size_bytes': session_metrics["total_size_bytes"]
                },
                source_paths=source_paths,
                destination_paths=copied_files and [item["destination"] for item in copied_files] or [],
                status='failed',
                error_details=str(e)
            )

    def process_item(self, session_id):
        """Process a single session.

        Args:
            session_id: Session ID to process

        Returns:
            Dict with processing result
        """
        try:
            # Calculate metrics for this session
            session_metrics = self.calculate_session_metrics(session_id)

            # If session qualifies, copy to curated location and record in DynamoDB
            if session_metrics["qualifies"]:
                result = self.copy_session_to_curated(session_metrics)
                return result or {"status": "skipped"}
            else:
                return {"status": "skipped"}
        except Exception as e:
            self.logger.error(f"Error processing session {session_id}: {e}")
            self.logger.debug("Error details:", exc_info=True)
            return {
                "status": "failed",
                "error": str(e)
            }

    @classmethod
    def add_subclass_arguments(cls, parser):
        """Add curation-specific command-line arguments.

        Args:
            parser: ArgumentParser instance
        """
        parser.add_argument('--source-prefix', type=str, default='data-collector/new-sessions/',
                          help='S3 prefix for source data')
        parser.add_argument('--dest-prefix', type=str, default='curated-h5/',
                          help='S3 prefix for destination data')
        parser.add_argument('--min-clean-duration', type=float, default=30.0,
                          help='Minimum clean duration in seconds')
        parser.add_argument('--min-data-rate', type=str, default='fnirs only',
                          choices=['fnirs only', 'both (fnirs+eeg)'],
                          help='Minimum data rate classification')
        parser.add_argument('--s3-bucket', type=str, default='conduit-data-dev',
                          help='S3 bucket name')

    @classmethod
    def from_args(cls, args):
        """Create a transform instance from command-line arguments.

        Args:
            args: Parsed command-line arguments

        Returns:
            Instance of CurationTransform
        """
        return cls(
            source_prefix=args.source_prefix,
            dest_prefix=args.dest_prefix,
            min_clean_duration_sec=args.min_clean_duration,
            min_data_rate_classification=args.min_data_rate,
            s3_bucket=args.s3_bucket,
            verbose=args.verbose,
            log_file=args.log_file,
            dry_run=args.dry_run
        )


# Entry point for running the transform from the command line
if __name__ == "__main__":
    CurationTransform.run_from_command_line()
