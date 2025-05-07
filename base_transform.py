"""
Base Transform class for data transformation pipeline stages.

This implements a new architecture with cleaner separation of concerns:
1. Base class manages all S3/AWS operations, session handling, and pipeline orchestration
2. Child classes only implement the business logic for processing a session
"""

import os
import sys
import boto3
import json
import time
import hashlib
import logging
import argparse
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Union, Callable, Tuple
from boto3.dynamodb.conditions import Key, Attr
from decimal import Decimal

try:
    # When running as an installed package
    from .utils.aws import (
        init_s3_client, init_dynamodb_resource, init_dynamodb_client,
        ensure_pipeline_table_exists, ensure_script_versions_table_exists,
        upload_script_with_md5_verification
    )
    from .utils.logging import setup_logging
except ImportError:
    # When running as a script
    from utils.aws import (
        init_s3_client, init_dynamodb_resource, init_dynamodb_client,
        ensure_pipeline_table_exists, ensure_script_versions_table_exists,
        upload_script_with_md5_verification
    )
    from utils.logging import setup_logging


class Session:
    """Represents a session (folder or file) in S3 with helper methods for operations.
    
    This abstraction simplifies working with files in a session by providing
    methods for listing, downloading, and copying files.
    """
    
    def __init__(self, s3_client, bucket, base_path, session_id, logger=None):
        """Initialize a session object.
        
        Args:
            s3_client: Boto3 S3 client
            bucket: S3 bucket name
            base_path: Base S3 path (prefix)
            session_id: Session identifier (plain ID, never a path)
            logger: Logger instance
        """
        self.s3 = s3_client
        self.bucket = bucket
        self.base_path = base_path
        self.session_id = session_id  # Store the plain session ID
        self.logger = logger or logging.getLogger(f"session.{session_id}")
        
        # Session is always a folder
        self.is_file = False
        self.file_path = None
        self.session_path = f"{base_path}{session_id}/"
        
        # Log the session path
        self.logger.debug(f"Session path: {self.session_path}")
        
        # Find path to the project root by locating the base_transform.py file
        base_transform_path = os.path.abspath(__file__)
        # The directory containing this file is the project root
        project_root = os.path.dirname(base_transform_path)
            
        # Create a timestamp-based workdir under transform_workdir in the project root
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        transform_workdir = os.path.join(project_root, 'transform_workdir')
        os.makedirs(transform_workdir, exist_ok=True)
        
        # Get the transform_id from the logger name or use 'unknown' if not available
        transform_id = 'unknown'
        if hasattr(self.logger, 'name') and '.' in self.logger.name:
            transform_id = self.logger.name.split('.')[-1]
        
        # Create the session-specific temp directory with new naming pattern
        session_workdir = os.path.join(transform_workdir, f'{transform_id}_{timestamp}_{session_id}')
        self.temp_dir = session_workdir
        os.makedirs(self.temp_dir, exist_ok=True)
        
        self.logger.debug(f"Created temp directory: {self.temp_dir}")
        
        self._file_cache = {}  # Cache for file listings
        
    def list_files(self, extension=None, subdir=None, prefix=None, use_cache=True):
        """List files directly in the session folder.
        
        Args:
            extension: Optional file extension filter (e.g., '.json', '.h5')
            subdir: Optional subdirectory within session (currently unused)
            prefix: Optional prefix filter (currently unused)
            use_cache: Whether to use cached file listings
            
        Returns:
            List of file keys
        """
        # Create cache key
        cache_key = f"{extension}"
        if use_cache and cache_key in self._file_cache:
            return self._file_cache[cache_key]
            
        # Only list files directly in the session folder
        # Using delimiter='/' ensures we only get top-level files, not those in subdirectories
        response = self.s3.list_objects_v2(
            Bucket=self.bucket,
            Prefix=self.session_path,
            Delimiter='/'
        )
        
        files = []
        if 'Contents' in response:
            for obj in response['Contents']:
                key = obj['Key']
                # Apply extension filter if specified
                if extension is None or key.endswith(extension):
                    files.append(key)
                    
        # Cache the results
        self._file_cache[cache_key] = files
        return files
        
    def get_file_sizes(self, extension=None):
        """Get sizes of files directly in the session folder.
        
        Args:
            extension: Optional file extension filter
            
        Returns:
            Dictionary mapping file keys to sizes in bytes
        """
        # Only get file sizes for files directly in the session folder
        # Using delimiter='/' ensures we only get top-level files
        response = self.s3.list_objects_v2(
            Bucket=self.bucket,
            Prefix=self.session_path,
            Delimiter='/'
        )
        
        file_sizes = {}
        if 'Contents' in response:
            for obj in response['Contents']:
                key = obj['Key']
                # Apply extension filter if specified
                if extension is None or key.endswith(extension):
                    file_sizes[key] = obj['Size']
                    
        return file_sizes
        
    def download_all(self):
        """Download all files to temp directory.
        
        Returns:
            Path to the temp directory
        """
        files = self.list_files()
        
        for file_key in files:
            self.download_file(file_key)
            
        return self.temp_dir
        
    def download_json_files(self):
        """Download only JSON files.
        
        Returns:
            Dict mapping file names to local paths
        """
        json_files = self.list_files(extension='.json')
        result = {}
        
        for file_key in json_files:
            local_path = self.download_file(file_key)
            file_name = os.path.basename(file_key)
            result[file_name] = local_path
            
        return result
        
    def download_file(self, file_key):
        """Download a specific file.
        
        Args:
            file_key: S3 key of the file
            
        Returns:
            Local path to the downloaded file
        """
        # Create the local path
        file_name = os.path.basename(file_key)
        local_path = os.path.join(self.temp_dir, file_name)
        
        # Download the file
        self.logger.debug(f"Downloading {file_key} to {local_path}")
        try:
            start_time = time.time()
            self.s3.download_file(self.bucket, file_key, local_path)
            duration = time.time() - start_time
            
            # Get file size
            file_size_mb = os.path.getsize(local_path) / (1024 * 1024)
            self.logger.debug(f"Downloaded {file_size_mb:.2f} MB in {duration:.2f} seconds")
            
            return local_path
        except Exception as e:
            self.logger.error(f"Error downloading {file_key}: {e}")
            raise
    
    def copy(self, source_key, destination_key):
        """Define a file copy operation (executed by the base class).
        
        Args:
            source_key: Source file key in S3
            destination_key: Destination file key in S3
            
        Returns:
            Dict with copy operation details
        """
        return {
            'source': source_key,
            'destination': destination_key
        }
        
    def create_upload_file(self, file_name, content=None):
        """Create a file for upload.
        
        Args:
            file_name: Name of the file to create
            content: Optional content to write to the file
            
        Returns:
            Local path to the created file
        """
        local_path = os.path.join(self.temp_dir, file_name)
        
        if content is not None:
            mode = 'w' if isinstance(content, str) else 'wb'
            with open(local_path, mode) as f:
                f.write(content)
                
        return local_path
        
    def save_dynamo_record(self, record, record_type="transform"):
        """Save a DynamoDB record locally as a JSON file.
        
        Args:
            record: The record to save (dictionary)
            record_type: Type of record (default: "transform")
            
        Returns:
            Local path to the created JSON file
        """
        # Create a records subdirectory if it doesn't exist
        records_dir = os.path.join(self.temp_dir, "dynamo_records")
        os.makedirs(records_dir, exist_ok=True)
        
        # Generate a filename based on record type and timestamp
        timestamp = record.get('timestamp', datetime.now().isoformat())
        record_id = record.get('data_id', 'unknown')
        transform_id = record.get('transform_id', 'unknown')
        
        # Create a safe filename
        safe_timestamp = timestamp.replace(':', '-').replace('.', '-')
        filename = f"{record_type}_{transform_id}_{record_id}_{safe_timestamp}.json"
        file_path = os.path.join(records_dir, filename)
        
        # Convert Decimal objects to floats for JSON serialization
        def decimal_to_float(obj):
            if isinstance(obj, Decimal):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: decimal_to_float(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [decimal_to_float(i) for i in obj]
            else:
                return obj
        
        # Convert record for JSON serialization
        serializable_record = decimal_to_float(record)
        
        # Write the record to a JSON file
        with open(file_path, 'w') as f:
            json.dump(serializable_record, f, indent=2)
        
        self.logger.debug(f"Saved DynamoDB record to {file_path}")
        return file_path
        
    def cleanup(self):
        """Clean up temp directory."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            self.logger.debug(f"Cleaned up {self.temp_dir}")


class BaseTransform:
    """Base class for all data transformation pipeline stages.
    
    This class provides a cleaner architecture with a clear separation
    of concerns:
    1. Base class handles S3 operations, session management, and pipeline flow
    2. Child classes implement only the business logic for processing a session
    
    Child classes MUST define:
    - SOURCE_PREFIX: Default source prefix for data
    - DEST_PREFIX: Default destination prefix for data
    """
    
    # These will be overridden by child classes
    SOURCE_PREFIX = None
    DEST_PREFIX = None
    
    def __init__(self, source_prefix: Optional[str] = None, 
                 destination_prefix: Optional[str] = None, 
                 transform_id: str = None, script_id: str = None, 
                 script_name: str = None, script_version: str = None, 
                 s3_bucket: str = 'conduit-data-dev', 
                 dynamodb_table: str = 'conduit-pipeline-metadata',
                 script_table: str = 'conduit-script-versions', 
                 script_prefix: str = 'scripts/',
                 verbose: bool = False, log_file: Optional[str] = None,
                 dry_run: bool = False, keep_local: bool = False):
        """Initialize a transform.
        
        Args:
            source_prefix: S3 prefix for source data (defaults to class.SOURCE_PREFIX)
            destination_prefix: S3 prefix for destination data (defaults to class.DEST_PREFIX)
            transform_id: Unique identifier for this transform
            script_id: Short identifier for the script
            script_name: Descriptive name for the script
            script_version: Version of the script
            s3_bucket: S3 bucket name
            dynamodb_table: DynamoDB table name for pipeline metadata
            script_table: DynamoDB table name for script versions
            script_prefix: S3 prefix for storing script versions
            verbose: Enable verbose logging
            log_file: Optional file path to write logs to
            dry_run: If True, simulate actions without making changes
            keep_local: If True, don't delete local working files after processing
        """
        # Check that subclass has defined prefixes
        if self.SOURCE_PREFIX is None:
            raise ValueError(f"{self.__class__.__name__} must define SOURCE_PREFIX class attribute")
        if self.DEST_PREFIX is None:
            raise ValueError(f"{self.__class__.__name__} must define DEST_PREFIX class attribute")
            
        # Use provided values or defaults from class
        self.source_prefix = source_prefix if source_prefix is not None else self.SOURCE_PREFIX
        self.destination_prefix = destination_prefix if destination_prefix is not None else self.DEST_PREFIX
        self.transform_id = transform_id
        self.script_id = script_id
        self.script_name = script_name
        self.script_version = script_version
        self.s3_bucket = s3_bucket
        self.dynamodb_table_name = dynamodb_table
        self.script_table_name = script_table
        self.script_prefix = script_prefix
        self.dry_run = dry_run
        self.keep_local = keep_local
        
        # Setup logging
        self.logger = setup_logging(
            f"pipeline.{self.transform_id}", 
            verbose=verbose, 
            log_file=log_file
        )
        
        # Initialize AWS clients
        self.logger.debug("Initializing AWS clients")
        self.s3 = init_s3_client()
        self.dynamodb = init_dynamodb_resource()
        self.dynamodb_client = init_dynamodb_client()
        
        # Ensure tables exist
        if not dry_run:
            self.logger.debug("Ensuring DynamoDB tables exist")
            ensure_pipeline_table_exists(self.dynamodb_client, self.dynamodb_table_name)
            ensure_script_versions_table_exists(self.dynamodb_client, self.script_table_name)
        
        # Upload script to S3 (or get existing path)
        self.logger.debug("Managing script version in S3")
        self.script_s3_path = self._upload_script_to_s3()
        
        # Get table reference
        self.table = self.dynamodb.Table(self.dynamodb_table_name)
        self.script_table = self.dynamodb.Table(self.script_table_name)
        
        # Log configuration
        self.logger.info(f"Transform initialized with:")
        self.logger.info(f"  Source prefix: {self.source_prefix}")
        self.logger.info(f"  Destination prefix: {self.destination_prefix}")
        self.logger.info(f"  Transform ID: {self.transform_id}")
        if self.dry_run and self.keep_local:
            self.logger.info(f"  TEST MODE ENABLED (dry-run + keep-local)")
        elif self.dry_run:
            self.logger.info(f"  DRY RUN MODE ENABLED")
        elif self.keep_local:
            self.logger.info(f"  KEEPING LOCAL FILES AND DYNAMO RECORDS (no cleanup)")
    
    def process_session(self, session: Session) -> Dict:
        """Process a single session.
        
        This method should be implemented by child classes to process
        a session and return the results.
        
        Args:
            session: Session object
            
        Returns:
            Dict with:
            - files_to_upload: List of (local_path, s3_key) tuples
            - files_to_copy: List of copy operation dicts
            - metadata: Dict of metadata to record
            - status: Success or failure status
        """
        raise NotImplementedError("Subclasses must implement process_session")
    
    def find_sessions(self):
        """Find sessions in the source prefix.
        
        Returns:
            List of session IDs to process
        """
        self.logger.info(f"Finding sessions in {self.source_prefix}")
        
        # First, check if there are files directly in the source_prefix
        response = self.s3.list_objects_v2(
            Bucket=self.s3_bucket,
            Prefix=self.source_prefix,
            Delimiter='/'
        )
        
        sessions = []
        
        # Handle files directly in the prefix
        if 'Contents' in response:
            for obj in response['Contents']:
                key = obj['Key']
                if key != self.source_prefix:  # Skip the prefix itself
                    file_name = os.path.basename(key)
                    if file_name.endswith('.h5') or file_name.endswith('.json'):
                        # The file itself is a session
                        sessions.append(file_name)
        
        # Handle directories (CommonPrefixes)
        if 'CommonPrefixes' in response:
            for prefix_obj in response['CommonPrefixes']:
                prefix = prefix_obj['Prefix']
                # Extract the session ID (last part of the prefix)
                session_id = prefix.rstrip('/').split('/')[-1]
                sessions.append(session_id)
        
        self.logger.info(f"Found {len(sessions)} sessions")
        return sessions
        
    def find_items_to_process(self):
        """Find items that need processing.
        
        This is a compatibility method for the CLI's interactive session selection.
        The base implementation simply returns the results of find_sessions().
        
        Returns:
            List of data IDs (session IDs or file paths) to process
        """
        return self.find_sessions()
    
    def _upload_script_to_s3(self):
        """Upload script to S3 if version doesn't exist or MD5 has changed."""
        # Get the script file path
        script_path = os.path.abspath(sys.modules[self.__class__.__module__].__file__)
        
        # Use the utility function for MD5-verified script uploads
        self.logger.debug(f"Managing script version {self.script_id} {self.script_version}")
        
        s3_path, was_uploaded = upload_script_with_md5_verification(
            s3_client=self.s3,
            dynamodb_resource=self.dynamodb,
            script_path=script_path,
            script_id=self.script_id,
            script_name=self.script_name,
            script_version=self.script_version,
            s3_bucket=self.s3_bucket,
            script_prefix=self.script_prefix,
            script_table_name=self.script_table_name,
            dry_run=self.dry_run
        )
        
        if was_uploaded:
            self.logger.info(f"Script uploaded to {s3_path}")
        else:
            self.logger.info(f"Using existing script at {s3_path}")
            
        return s3_path
    
    def get_processed_items(self, include_skipped=True):
        """Get data IDs already processed by this transform.
        
        Args:
            include_skipped: Whether to include skipped items
            
        Returns:
            Set of processed data IDs
        """
        processed_items = set()
        try:
            self.logger.debug(f"Querying processed items for transform {self.transform_id}")
            
            # Build filter expression based on whether to include skipped items
            if include_skipped:
                # Filter for both 'success' and 'skipped' status
                filter_expr = Attr('status').is_in(['success', 'skipped'])
            else:
                # Only include 'success' status
                filter_expr = Attr('status').eq('success')
            
            # Initial query
            response = self.table.query(
                IndexName='TransformIndex',
                KeyConditionExpression=Key('transform_id').eq(self.transform_id),
                FilterExpression=filter_expr,
                ProjectionExpression="data_id"
            )
            
            for item in response.get('Items', []):
                processed_items.add(item['data_id'])
            
            # Handle pagination
            while 'LastEvaluatedKey' in response:
                response = self.table.query(
                    IndexName='TransformIndex',
                    KeyConditionExpression=Key('transform_id').eq(self.transform_id),
                    FilterExpression=filter_expr,
                    ProjectionExpression="data_id",
                    ExclusiveStartKey=response['LastEvaluatedKey']
                )
                for item in response.get('Items', []):
                    processed_items.add(item['data_id'])
            
            # Create message based on what we're including
            status_msg = "processed or skipped" if include_skipped else "successfully processed"
            self.logger.info(f"Found {len(processed_items)} already {status_msg} items")
            return processed_items
        except Exception as e:
            self.logger.error(f"Error getting processed items: {e}")
            return set()
    
    def _convert_floats_to_decimal(self, obj):
        """Convert all float values in an object to Decimal for DynamoDB compatibility."""
        if isinstance(obj, float):
            return Decimal(str(obj))
        elif isinstance(obj, dict):
            return {k: self._convert_floats_to_decimal(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_floats_to_decimal(i) for i in obj]
        else:
            return obj
    
    def record_transform(self, data_id: str, transform_metadata: Dict[str, Any], 
                         source_paths: Optional[List[str]] = None, 
                         destination_paths: Optional[List[str]] = None,
                         status: str = 'success', 
                         error_details: Optional[str] = None,
                         parent_transforms: Optional[List[str]] = None,
                         session: Optional['Session'] = None):
        """Record a transform in DynamoDB.
        
        Args:
            data_id: Data ID being processed
            transform_metadata: Dict of transform-specific metadata
            source_paths: List of source paths
            destination_paths: List of destination paths
            status: Status of the transform ('success', 'failed', or 'skipped')
            error_details: Error details if status is 'failed'
            parent_transforms: List of prerequisite transform IDs
            session: Session object for local record saving (optional)
            
        Returns:
            The created record or None if operation failed
        """
        if parent_transforms is None:
            parent_transforms = []
            
        timestamp = datetime.now().isoformat()
        
        # Convert all float values in transform_metadata to Decimal for DynamoDB compatibility
        converted_metadata = self._convert_floats_to_decimal(transform_metadata)
        
        record = {
            'data_id': data_id,
            'transform_id': self.transform_id,
            'timestamp': timestamp,
            'script_info': {
                'id': self.script_id,
                'name': self.script_name,
                'version': self.script_version,
                's3_path': self.script_s3_path
            },
            'transform_metadata': converted_metadata,
            'status': status,
            'parent_transforms': parent_transforms,
            'next_transforms': []
        }
        
        if source_paths:
            record['source_paths'] = source_paths
            
        if destination_paths:
            record['destination_paths'] = destination_paths
            
        if error_details:
            record['error_details'] = error_details
        
        # Save record locally if keep_local is True and session is provided
        # Do this regardless of dry-run status, but with a standardized record type
        if self.keep_local and session is not None:
            try:
                record_type = "transform"
                local_record_path = session.save_dynamo_record(record, record_type)
                if self.dry_run:
                    self.logger.info(f"[DRY RUN] Saved transform record locally to {local_record_path}")
                else:
                    self.logger.info(f"Saved transform record locally to {local_record_path}")
            except Exception as local_save_error:
                self.logger.warning(f"Failed to save record locally: {local_save_error}")
        
        # Continue with normal dry-run handling    
        if self.dry_run:
            self.logger.info(f"[DRY RUN] Would record transform for {data_id} with status '{status}'")
            self.logger.debug(f"[DRY RUN] Record that would be created: {record}")
                    
            return record
            
        # Store the record
        try:
            self.logger.debug(f"Storing record for {data_id} with status '{status}'")
            self.table.put_item(Item=record)
            return record
        except Exception as e:
            self.logger.error(f"Error recording transform: {e}")
            return None
    
    def process_item(self, session_id: str):
        """Process a single session item.
        
        Args:
            session_id: Session ID to process
            
        Returns:
            Dict with processing result
        """
        try:
            # Log keep_local value before processing
            self.logger.info(f"BaseTransform.process_item: keep_local before processing = {self.keep_local}")
            self.logger.info(f"BaseTransform.process_item: dry_run before processing = {self.dry_run}")
            
            # Create a session object
            session = Session(
                s3_client=self.s3,
                bucket=self.s3_bucket,
                base_path=self.source_prefix,
                session_id=session_id,
                logger=self.logger
            )
            
            # Call the child class implementation - always runs in normal mode with actual processing
            # Dry run will be handled in the base class after process_session returns
            self.logger.info(f"About to call process_session with keep_local = {self.keep_local}")
            result = self.process_session(session)
            self.logger.info(f"After process_session call, keep_local = {self.keep_local}")
            
            # Extract the results
            files_to_upload = result.get('files_to_upload', [])
            files_to_copy = result.get('files_to_copy', [])
            metadata = result.get('metadata', {})
            status = result.get('status', 'success')
            error_details = result.get('error_details')
            
            # Process uploads (if any)
            destination_paths = []
            source_paths = []
            
            # Track source paths from copy operations
            for copy_op in files_to_copy:
                source_key = copy_op['source']
                dest_key = copy_op['destination']
                source_path = f"s3://{self.s3_bucket}/{source_key}"
                dest_path = f"s3://{self.s3_bucket}/{dest_key}"
                
                source_paths.append(source_path)
                destination_paths.append(dest_path)
                
                if not self.dry_run:
                    self.logger.info(f"Copying {source_path} to {dest_path}")
                    self.s3.copy_object(
                        Bucket=self.s3_bucket,
                        CopySource={'Bucket': self.s3_bucket, 'Key': source_key},
                        Key=dest_key
                    )
                else:
                    self.logger.info(f"[DRY RUN] Would copy {source_path} to {dest_path}")
            
            # Process upload files
            for local_path, s3_key in files_to_upload:
                dest_path = f"s3://{self.s3_bucket}/{s3_key}"
                destination_paths.append(dest_path)
                
                if not self.dry_run:
                    self.logger.info(f"Uploading {local_path} to {dest_path}")
                    self.s3.upload_file(local_path, self.s3_bucket, s3_key)
                else:
                    self.logger.info(f"[DRY RUN] Would upload {local_path} to {dest_path}")
            
            # Record the transform
            transform_record = self.record_transform(
                data_id=session.session_id,
                transform_metadata=metadata,
                source_paths=source_paths,
                destination_paths=destination_paths,
                status=status,
                error_details=error_details,
                session=session  # Pass the session object for local record saving
            )
            
            # Log keep_local right before cleanup decision
            self.logger.info(f"keep_local before cleanup decision: {self.keep_local}")
            
            # Clean up unless keep_local is set
            if not self.keep_local:
                self.logger.info(f"Cleaning up temp files since keep_local = {self.keep_local}")
                session.cleanup()
            else:
                self.logger.info(f"Keeping local files and records in: {session.temp_dir}")
                # Print to stdout as well so it's clearly visible to users
                print(f"\nTemporary files and DynamoDB records preserved for inspection in:\n{session.temp_dir}")
                print(f"DynamoDB records saved in: {session.temp_dir}/dynamo_records/\n")
            
            return transform_record
            
        except Exception as e:
            self.logger.error(f"Error processing session {session_id}: {e}")
            self.logger.debug("Error details:", exc_info=True)
            
            # Log keep_local value in exception handling
            self.logger.info(f"keep_local in exception handling: {self.keep_local}")
            
            # Clean up in case of error, unless keep_local is set
            if not self.keep_local:
                try:
                    if 'session' in locals():  # Check if session was created before the error
                        self.logger.info(f"Cleaning up temp files due to exception (keep_local = {self.keep_local})")
                        session.cleanup()
                except Exception as cleanup_error:
                    self.logger.warning(f"Error during cleanup: {cleanup_error}")
            else:
                # Log the location of the kept files
                if 'session' in locals() and hasattr(session, 'temp_dir'):
                    self.logger.info(f"Keeping local files and records despite error in: {session.temp_dir}")
                    # Print to stdout as well so it's clearly visible to users
                    print(f"\nTemporary files and DynamoDB records preserved for inspection (despite error) in:\n{session.temp_dir}")
                    print(f"DynamoDB records saved in: {session.temp_dir}/dynamo_records/\n")
                
            # Record the failure
            try:
                return self.record_transform(
                    data_id=session_id,
                    transform_metadata={},
                    status='failed',
                    error_details=str(e),
                    session=session if 'session' in locals() else None  # Pass session if it exists
                )
            except Exception:
                return {"status": "failed", "error": str(e)}
    
    def process_items_batch(self, session_ids: List[str], start_idx: int = 0, end_idx: Optional[int] = None):
        """Process a batch of items.
        
        Args:
            session_ids: List of session IDs to process
            start_idx: Starting index
            end_idx: Ending index (exclusive)
            
        Returns:
            Dict with batch processing statistics
        """
        if end_idx is None:
            end_idx = len(session_ids)
            
        batch_size = end_idx - start_idx
        batch_ids = session_ids[start_idx:end_idx]
        
        if self.dry_run:
            self.logger.info(f"[DRY RUN] Processing batch of {batch_size} items (indices {start_idx}-{end_idx-1})")
        else:
            self.logger.info(f"Processing batch of {batch_size} items (indices {start_idx}-{end_idx-1})")
        
        stats = {
            "processed": 0,
            "success": 0,
            "failed": 0,
            "skipped": 0
        }
        
        for i, session_id in enumerate(batch_ids, 1):
            self.logger.info(f"Processing item {i}/{batch_size}: {session_id}")
            
            try:
                result = self.process_item(session_id)
                stats["processed"] += 1
                
                if result and 'status' in result:
                    if result['status'] == 'success':
                        stats["success"] += 1
                    elif result['status'] == 'failed':
                        stats["failed"] += 1
                    elif result['status'] == 'skipped':
                        stats["skipped"] += 1
            except Exception as e:
                self.logger.error(f"Error processing item {session_id}: {e}")
                self.logger.debug("Error details:", exc_info=True)
                stats["failed"] += 1
        
        self.logger.info(f"Batch statistics: {stats}")
        return stats
    
    def run_pipeline(self, session_ids: Optional[List[str]] = None, batch_size: int = 0, 
                    max_items: int = 0, start_idx: int = 0, include_processed: bool = False):
        """Run the pipeline on specified session IDs or find new sessions to process.
        
        Args:
            session_ids: List of session IDs (if None, will find sessions)
            batch_size: Number of items to process in each batch
            max_items: Maximum number of items to process
            start_idx: Starting index for processing
            include_processed: Whether to include already processed sessions
            
        Returns:
            Dict with processing statistics
        """
        if session_ids is None:
            # Find sessions that need processing
            self.logger.info("Finding sessions to process")
            session_ids = self.find_sessions()
            
        # Only filter out processed sessions if include_processed is False
        unprocessed_ids = session_ids
        if not include_processed:
            # Get processed items (including those with 'skipped' status)
            processed_items = self.get_processed_items(include_skipped=True)
            unprocessed_ids = [id for id in session_ids if id not in processed_items]
            self.logger.info(f"Found {len(unprocessed_ids)} unprocessed items out of {len(session_ids)} total")
        else:
            self.logger.info(f"Processing all {len(session_ids)} items including previously processed ones")
        
        # Apply max_items limit if specified
        if max_items > 0 and max_items < len(unprocessed_ids):
            self.logger.info(f"Limiting to {max_items} items as requested")
            unprocessed_ids = unprocessed_ids[:max_items]
            
        # Log dry run mode if enabled
        if self.dry_run:
            self.logger.info("=== DRY RUN MODE ENABLED - No files will be modified, no DynamoDB records will be created ===")
            
        # Process in batches if batch_size is specified
        if batch_size > 0:
            self.logger.info(f"Processing in batches of {batch_size} items")
            total_stats = {
                "processed": 0,
                "success": 0,
                "failed": 0,
                "skipped": 0
            }
            
            batch_start = start_idx
            batch_count = 0
            
            while batch_start < len(unprocessed_ids):
                batch_count += 1
                batch_end = min(batch_start + batch_size, len(unprocessed_ids))
                
                self.logger.info(f"Starting batch {batch_count} (items {batch_start}-{batch_end-1})")
                batch_stats = self.process_items_batch(
                    unprocessed_ids, start_idx=batch_start, end_idx=batch_end
                )
                
                # Update total stats
                for key in total_stats:
                    total_stats[key] += batch_stats.get(key, 0)
                    
                batch_start = batch_end
                
            return total_stats
        else:
            # Process all at once
            self.logger.info(f"Processing all {len(unprocessed_ids)} items at once")
            return self.process_items_batch(unprocessed_ids)
    
    @classmethod
    def add_arguments(cls, parser):
        """Add standard command-line arguments for the transform.
        
        Args:
            parser: ArgumentParser instance
        """
        parser.add_argument('--source-prefix', type=str, default=cls.SOURCE_PREFIX,
                          help=f'S3 prefix for source data (default: {cls.SOURCE_PREFIX})')
        parser.add_argument('--dest-prefix', type=str, default=cls.DEST_PREFIX,
                          help=f'S3 prefix for destination data (default: {cls.DEST_PREFIX})')
        parser.add_argument('--s3-bucket', type=str, default='conduit-data-dev',
                          help='S3 bucket name')
        parser.add_argument('--batch-size', type=int, default=0, 
                          help='Number of items to process in each batch (0 for all at once)')
        parser.add_argument('--start-idx', type=int, default=0,
                          help='Starting index for processing')
        parser.add_argument('--max-items', type=int, default=0,
                          help='Maximum number of items to process (0 for no limit)')
        parser.add_argument('--verbose', '-v', action='store_true',
                          help='Enable verbose logging (DEBUG level)')
        parser.add_argument('--log-file', type=str, default=None,
                          help='Path to write log output to a file')
        parser.add_argument('--dry-run', '-n', action='store_true',
                          help="Dry run mode - don't actually modify files or DynamoDB")
        parser.add_argument('--keep-local', '-k', action='store_true',
                          help="Keep local temporary files and DynamoDB records (don't clean up the working directory)")
        parser.add_argument('--test', '-t', action='store_true',
                          help="Test mode - combines both --dry-run and --keep-local flags")
        parser.add_argument('--reset', '-r', action='store_true',
                          help="Reset the transform by deleting its entries from DynamoDB")
        # Create a mutually exclusive group for session filtering options
        session_filter_group = parser.add_mutually_exclusive_group()
        session_filter_group.add_argument('--include-processed', action='store_true',
                          help='Include and reprocess sessions that have already been processed')
        session_filter_group.add_argument('--include-skipped', action='store_true',
                          help='Include and reprocess sessions that were previously skipped')
        session_filter_group.add_argument('--skipped-only', action='store_true',
                          help='Only process sessions that were previously skipped')
    
    @classmethod
    def add_subclass_arguments(cls, parser):
        """Add subclass-specific command-line arguments.
        
        Args:
            parser: ArgumentParser instance
            
        This method should be overridden by subclasses to add their own arguments.
        """
        pass
    
    @classmethod
    def from_args(cls, args):
        """Create a transform instance from command-line arguments.
        
        Args:
            args: Parsed command-line arguments
            
        Returns:
            Instance of the transform class
            
        This method should be overridden by subclasses.
        
        Note that all args from add_arguments() are included, such as:
        - dry_run: Flag for dry run mode
        - keep_local: Flag to keep local files
        """
        raise NotImplementedError("Subclasses must implement from_args")
    
    def reset_transform_data(self):
        """Reset the transform by deleting all its entries from DynamoDB.
        
        This method deletes all records for this transform_id from the pipeline metadata table.
        It provides a clean slate for reprocessing.
        
        Returns:
            Dict with information about the reset operation
        """
        self.logger.warning(f"Resetting all data for transform {self.transform_id}")
        
        try:
            # Get a list of items to delete
            items_to_delete = []
            response = self.table.query(
                IndexName='TransformIndex',
                KeyConditionExpression=Key('transform_id').eq(self.transform_id),
                ProjectionExpression="data_id"
            )
            
            for item in response.get('Items', []):
                items_to_delete.append(item['data_id'])
                
            # Handle pagination
            while 'LastEvaluatedKey' in response:
                response = self.table.query(
                    IndexName='TransformIndex',
                    KeyConditionExpression=Key('transform_id').eq(self.transform_id),
                    ProjectionExpression="data_id",
                    ExclusiveStartKey=response['LastEvaluatedKey']
                )
                for item in response.get('Items', []):
                    items_to_delete.append(item['data_id'])
            
            item_count = len(items_to_delete)
            self.logger.info(f"Found {item_count} items to delete for transform {self.transform_id}")
            
            if item_count == 0:
                return {
                    "status": "success", 
                    "message": f"No items found for transform {self.transform_id}",
                    "deleted_count": 0
                }
                
            # In dry run mode, we don't actually delete anything
            if self.dry_run:
                self.logger.info(f"[DRY RUN] Would delete {item_count} items for transform {self.transform_id}")
                return {
                    "status": "dry_run", 
                    "message": f"Would delete {item_count} items for transform {self.transform_id}",
                    "would_delete_count": item_count
                }
                
            # Delete items in batches of 25 (DynamoDB batch limit)
            batch_size = 25
            deleted_count = 0
            for i in range(0, item_count, batch_size):
                batch = items_to_delete[i:i+batch_size]
                batch_delete_requests = []
                
                for data_id in batch:
                    batch_delete_requests.append({
                        'DeleteRequest': {
                            'Key': {
                                'data_id': data_id,
                                'transform_id': self.transform_id
                            }
                        }
                    })
                
                if batch_delete_requests:
                    self.logger.debug(f"Deleting batch of {len(batch_delete_requests)} items")
                    self.dynamodb.batch_write_item(
                        RequestItems={
                            self.dynamodb_table_name: batch_delete_requests
                        }
                    )
                    deleted_count += len(batch_delete_requests)
                    
            self.logger.info(f"Successfully deleted {deleted_count} items for transform {self.transform_id}")
            return {
                "status": "success", 
                "message": f"Successfully deleted {deleted_count} items for transform {self.transform_id}",
                "deleted_count": deleted_count
            }
        except Exception as e:
            self.logger.error(f"Error resetting transform data: {e}")
            return {
                "status": "error", 
                "message": f"Error resetting transform data: {e}"
            }
    
    @classmethod
    def run_from_command_line(cls):
        """Run the transform from the command line."""
        # Create argument parser
        parser = argparse.ArgumentParser(description=cls.__doc__)
        
        # Add standard arguments
        cls.add_arguments(parser)
        
        # Add subclass-specific arguments
        cls.add_subclass_arguments(parser)
        
        # Parse arguments
        args = parser.parse_args()
        
        # Handle the --test flag (combines dry-run and keep-local)
        if hasattr(args, 'test') and args.test:
            args.dry_run = True
            args.keep_local = True
        
        # Create transform instance
        transform = cls.from_args(args)
        
        # Handle the --reset flag
        if hasattr(args, 'reset') and args.reset:
            # Confirm before proceeding
            if not args.dry_run:  # Only ask for confirmation if not in dry-run mode
                confirm = input(f"Are you sure you want to delete all data for transform '{transform.transform_id}'? (y/N): ")
                if confirm.lower() != 'y':
                    transform.logger.info("Reset operation cancelled by user")
                    return
                    
            # Reset the transform data
            result = transform.reset_transform_data()
            
            # In dry run mode, the reset operation won't actually delete anything
            # but will show what would be deleted
            if args.dry_run:
                transform.logger.info(f"[DRY RUN] {result['message']}")
                if 'would_delete_count' in result and result['would_delete_count'] > 0:
                    transform.logger.info(f"[DRY RUN] Use --reset without --dry-run to actually delete the data")
            else:
                transform.logger.info(result['message'])
                
            # Exit after reset
            return
        
        # Run the pipeline
        transform.run_pipeline(
            batch_size=args.batch_size,
            max_items=args.max_items,
            start_idx=args.start_idx,
            include_processed=hasattr(args, 'include_processed') and args.include_processed
        )