"""
Base class for data transformation pipeline stages.
"""

import os
import sys
import boto3
import json
import time
import hashlib
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Union, Callable
from boto3.dynamodb.conditions import Key
from decimal import Decimal

try:
    # When running as an installed package
    from .utils.aws import (
        init_s3_client, init_dynamodb_resource, init_dynamodb_client,
        ensure_pipeline_table_exists, ensure_script_versions_table_exists
    )
    from .utils.logging import setup_logging
except ImportError:
    # When running as a script
    from utils.aws import (
        init_s3_client, init_dynamodb_resource, init_dynamodb_client,
        ensure_pipeline_table_exists, ensure_script_versions_table_exists
    )
    from utils.logging import setup_logging


class DataTransform:
    """Base class for all data transformation pipeline stages.
    
    This class provides the foundation for creating data processing pipelines
    where each stage performs a specific transformation on the data and records
    its actions in a consistent, traceable manner.
    
    Key features:
    - Tracks processed items to avoid reprocessing
    - Records lineage by tracking dependencies between transforms
    - Supports dry-run mode for testing without making actual changes
    - Implements batching for large datasets
    - Saves script versions for reproducibility
    - Common utilities for S3 operations and file handling
    """
    
    def __init__(self, transform_id: str, script_id: str, script_name: str, script_version: str, 
                 s3_bucket: str, dynamodb_table: str = 'conduit-pipeline-metadata',
                 script_table: str = 'conduit-script-versions', 
                 script_prefix: str = 'scripts/',
                 verbose: bool = False, log_file: Optional[str] = None,
                 dry_run: bool = False):
        """Initialize a data transformation pipeline stage.
        
        Args:
            transform_id: Unique identifier for this transform (e.g., "curation_v1")
            script_id: Short identifier for the script (e.g., "0A")
            script_name: Descriptive name for the script (e.g., "curate_sessions")
            script_version: Version of the script (e.g., "v0")
            s3_bucket: S3 bucket name for storing data
            dynamodb_table: DynamoDB table name for pipeline metadata
            script_table: DynamoDB table name for script versions
            script_prefix: S3 prefix for storing script versions
            verbose: Enable verbose logging (DEBUG level)
            log_file: Optional file path to write logs to
            dry_run: If True, simulate actions without making changes
        """
        self.transform_id = transform_id
        self.script_id = script_id
        self.script_name = script_name
        self.script_version = script_version
        self.s3_bucket = s3_bucket
        self.dynamodb_table_name = dynamodb_table
        self.script_table_name = script_table
        self.script_prefix = script_prefix
        self.dry_run = dry_run
        
        # Setup logging
        self.logger = setup_logging(
            f"pipeline.{self.transform_id}", 
            verbose=verbose, 
            log_file=log_file
        )
        
        # Initialize AWS clients
        self.logger.debug("Initializing AWS clients")
        self.s3 = self._init_s3_client()
        self.dynamodb = self._init_dynamodb_resource()
        self.dynamodb_client = self._init_dynamodb_client()
        
        # Ensure tables exist
        if not dry_run:
            self.logger.debug("Ensuring DynamoDB tables exist")
            self._ensure_pipeline_table_exists()
            self._ensure_script_versions_table_exists()
        
        # Upload script to S3 (or get existing path)
        self.logger.debug("Managing script version in S3")
        self.script_s3_path = self._upload_script_to_s3()
        
        # Get table reference
        self.table = self.dynamodb.Table(self.dynamodb_table_name)
        self.script_table = self.dynamodb.Table(self.script_table_name)
    
    def _init_s3_client(self):
        """Initialize S3 client."""
        return init_s3_client()
    
    def _init_dynamodb_resource(self):
        """Initialize DynamoDB resource."""
        return init_dynamodb_resource()
    
    def _init_dynamodb_client(self):
        """Initialize DynamoDB client."""
        return init_dynamodb_client()
    
    def _ensure_pipeline_table_exists(self):
        """Ensure pipeline metadata table exists."""
        ensure_pipeline_table_exists(self.dynamodb_client, self.dynamodb_table_name)
    
    def _ensure_script_versions_table_exists(self):
        """Ensure script versions table exists."""
        ensure_script_versions_table_exists(self.dynamodb_client, self.script_table_name)
    
    # S3 Operations
    def list_s3_objects(self, prefix: str, filter_func: Optional[Callable[[str], bool]] = None, 
                      delimiter: Optional[str] = None) -> List[str]:
        """List S3 objects with optional filtering.

        Args:
            prefix: S3 prefix to list
            filter_func: Optional filter function to apply to objects
            delimiter: Optional delimiter for listing directories

        Returns:
            List of matching objects or prefixes
        """
        result = []
        paginator = self.s3.get_paginator('list_objects_v2')
        
        paginate_args = {
            'Bucket': self.s3_bucket,
            'Prefix': prefix
        }
        
        if delimiter:
            paginate_args['Delimiter'] = delimiter
        
        for page in paginator.paginate(**paginate_args):
            # Process directory prefixes if delimiter is used
            if delimiter and 'CommonPrefixes' in page:
                for prefix_obj in page['CommonPrefixes']:
                    prefix_val = prefix_obj['Prefix']
                    if not filter_func or filter_func(prefix_val):
                        result.append(prefix_val)
            
            # Process file objects
            if 'Contents' in page:
                for obj in page['Contents']:
                    key = obj['Key']
                    if not filter_func or filter_func(key):
                        result.append(key)
        
        return result
    
    def create_temp_path(self, filename: str) -> str:
        """Create a temporary file path.
        
        Args:
            filename: Base filename
            
        Returns:
            Temporary file path
        """
        # Make sure we only use the base filename
        return os.path.join('/tmp', os.path.basename(filename))
    
    def cleanup_temp_file(self, path: str) -> bool:
        """Safely remove a temporary file if it exists.
        
        Args:
            path: Path to the file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if os.path.exists(path):
                os.remove(path)
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error cleaning up temporary file {path}: {e}")
            return False
    
    def get_s3_uri(self, key: str) -> str:
        """Create a full S3 URI from a key.
        
        Args:
            key: S3 object key
            
        Returns:
            Full S3 URI with bucket and key
        """
        return f"s3://{self.s3_bucket}/{key}"
    
    def download_s3_file(self, s3_key: str, local_path: Optional[str] = None) -> str:
        """Download a file from S3 with timing and error handling.
        
        Args:
            s3_key: S3 key to download
            local_path: Optional local path (default: /tmp/filename)
            
        Returns:
            Path to local file
        """
        if not local_path:
            local_path = self.create_temp_path(s3_key)
        
        try:
            self.logger.info(f"Downloading {s3_key} to {local_path}")
            
            start_time = time.time()
            self.s3.download_file(self.s3_bucket, s3_key, local_path)
            download_time = time.time() - start_time
            
            file_size_mb = os.path.getsize(local_path) / (1024 * 1024)
            self.logger.info(f"Downloaded {file_size_mb:.2f} MB in {download_time:.2f} seconds")
            
            if self.dry_run:
                self.logger.info(f"[DRY RUN] Downloaded {s3_key} for inspection only")
            
            return local_path
        except Exception as e:
            self.logger.error(f"Error downloading file {s3_key}: {e}")
            if self.dry_run:
                self.logger.warning(f"[DRY RUN] Would fail with download error: {e}")
                # In dry run, create an empty file for error handling
                with open(local_path, 'wb') as f:
                    f.write(b'')
                return local_path
            raise
    
    def upload_s3_file(self, local_path: str, s3_key: str) -> str:
        """Upload a file to S3 with timing and error handling.
        
        Args:
            local_path: Local file path
            s3_key: S3 key to upload to
            
        Returns:
            S3 URI of uploaded file
        """
        dest_path = self.get_s3_uri(s3_key)
        
        try:
            self.logger.info(f"Uploading file to {dest_path}")
            
            if self.dry_run:
                self.logger.info(f"[DRY RUN] Would upload {local_path} to {dest_path}")
                return dest_path
            
            upload_start = time.time()
            self.s3.upload_file(local_path, self.s3_bucket, s3_key)
            upload_time = time.time() - upload_start
            
            file_size_mb = os.path.getsize(local_path) / (1024 * 1024)
            self.logger.info(f"Uploaded {file_size_mb:.2f} MB in {upload_time:.2f} seconds")
            
            return dest_path
        except Exception as e:
            self.logger.error(f"Error uploading file {local_path} to {s3_key}: {e}")
            raise
    
    def copy_s3_object(self, source_key: str, dest_key: str) -> str:
        """Copy an object within S3 with proper error handling.
        
        Args:
            source_key: Source object key
            dest_key: Destination object key
            
        Returns:
            S3 URI of the destination object
        """
        dest_path = self.get_s3_uri(dest_key)
        source_path = self.get_s3_uri(source_key)
        
        try:
            self.logger.info(f"Copying {source_path} to {dest_path}")
            
            if self.dry_run:
                self.logger.info(f"[DRY RUN] Would copy {source_path} to {dest_path}")
                return dest_path
            
            # Use copy_object for efficiency (stays server-side in S3)
            self.s3.copy_object(
                Bucket=self.s3_bucket,
                CopySource={'Bucket': self.s3_bucket, 'Key': source_key},
                Key=dest_key
            )
            
            self.logger.info(f"Successfully copied object to {dest_path}")
            return dest_path
        except Exception as e:
            self.logger.error(f"Error copying S3 object: {e}")
            raise
    
    def _upload_script_to_s3(self):
        """Upload script to S3 only if version doesn't exist or MD5 has changed, and return its path."""
        try:
            # When running as an installed package
            from .utils.aws import upload_script_with_md5_verification
        except ImportError:
            # When running as a script
            from utils.aws import upload_script_with_md5_verification
        
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
    
    def get_processed_items(self):
        """Get data IDs already processed by this transform.
        
        Returns:
            Set of data IDs (e.g., session IDs)
        """
        processed_items = set()
        try:
            self.logger.debug(f"Querying processed items for transform {self.transform_id}")
            response = self.table.query(
                IndexName='TransformIndex',
                KeyConditionExpression=Key('transform_id').eq(self.transform_id),
                ProjectionExpression="data_id"
            )
            
            for item in response.get('Items', []):
                processed_items.add(item['data_id'])
            
            # Handle pagination
            while 'LastEvaluatedKey' in response:
                response = self.table.query(
                    IndexName='TransformIndex',
                    KeyConditionExpression=Key('transform_id').eq(self.transform_id),
                    ProjectionExpression="data_id",
                    ExclusiveStartKey=response['LastEvaluatedKey']
                )
                for item in response.get('Items', []):
                    processed_items.add(item['data_id'])
                    
            self.logger.info(f"Found {len(processed_items)} already processed items")
            return processed_items
        except Exception as e:
            self.logger.error(f"Error getting processed items: {e}")
            return set()
    
    def get_prerequisite_transform_records(self, data_id: str, transform_id: str):
        """Get records for a prerequisite transform.
        
        Args:
            data_id: Data ID to check
            transform_id: Transform ID to check
            
        Returns:
            List of records for the specified transform
        """
        try:
            self.logger.debug(f"Getting prerequisite records for {data_id}, transform {transform_id}")
            response = self.table.query(
                KeyConditionExpression=Key('data_id').eq(data_id) & 
                                      Key('transform_id').eq(transform_id)
            )
            return response.get('Items', [])
        except Exception as e:
            self.logger.error(f"Error getting prerequisite records: {e}")
            return []
    
    def _convert_floats_to_decimal(self, obj):
        """Convert all float values in an object to Decimal for DynamoDB compatibility.
        
        Args:
            obj: The object (dict, list, float, etc.) to convert
            
        Returns:
            The object with all floats converted to Decimal
        """
        if isinstance(obj, float):
            return Decimal(str(obj))  # Convert float to Decimal using string representation for precision
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
                         parent_transforms: Optional[List[str]] = None):
        """Record a transform in DynamoDB.
        
        Args:
            data_id: Data ID being processed (e.g., session ID)
            transform_metadata: Dict of transform-specific metadata
            source_paths: List of source paths
            destination_paths: List of destination paths
            status: Status of the transform ('success', 'failed', or 'skipped')
            error_details: Error details if status is 'failed'
            parent_transforms: List of prerequisite transform IDs
            
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
            
        if self.dry_run:
            self.logger.info(f"[DRY RUN] Would record transform for {data_id} with status '{status}'")
            self.logger.debug(f"[DRY RUN] Record that would be created: {record}")
            return record
            
        # Update parent records to include this transform as next_transform
        for parent_id in parent_transforms:
            try:
                parent_records = self.get_prerequisite_transform_records(data_id, parent_id)
                for parent_record in parent_records:
                    next_transforms = parent_record.get('next_transforms', [])
                    if self.transform_id not in next_transforms:
                        next_transforms.append(self.transform_id)
                        self.table.update_item(
                            Key={
                                'data_id': data_id,
                                'transform_id': parent_id
                            },
                            UpdateExpression="set next_transforms = :n",
                            ExpressionAttributeValues={
                                ':n': next_transforms
                            }
                        )
            except Exception as e:
                self.logger.warning(f"Failed to update parent transform {parent_id}: {e}")
        
        # Store the record
        try:
            self.logger.debug(f"Storing record for {data_id} with status '{status}'")
            self.table.put_item(Item=record)
            return record
        except Exception as e:
            self.logger.error(f"Error recording transform: {e}")
            return None
    
    def process_item(self, data_id: str):
        """Process a single item.
        
        Args:
            data_id: ID of the item to process
            
        Returns:
            Dict with processing result
            
        Raises:
            NotImplementedError: Subclasses must implement this method
        """
        raise NotImplementedError("Subclasses must implement process_item")
    
    def process_items_batch(self, data_ids: List[str], start_idx: int = 0, end_idx: Optional[int] = None):
        """Process a batch of items.
        
        Args:
            data_ids: List of data IDs to process
            start_idx: Starting index in the data_ids list
            end_idx: Ending index (exclusive) in the data_ids list
            
        Returns:
            Dict with batch processing statistics
        """
        if end_idx is None:
            end_idx = len(data_ids)
            
        batch_size = end_idx - start_idx
        batch_ids = data_ids[start_idx:end_idx]
        
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
        
        for i, data_id in enumerate(batch_ids, 1):
            self.logger.info(f"Processing item {i}/{batch_size}: {data_id}")
            
            try:
                result = self.process_item(data_id)
                stats["processed"] += 1
                
                if result and 'status' in result:
                    if result['status'] == 'success':
                        stats["success"] += 1
                    elif result['status'] == 'failed':
                        stats["failed"] += 1
                    elif result['status'] == 'skipped':
                        stats["skipped"] += 1
            except Exception as e:
                self.logger.error(f"Error processing item {data_id}: {e}")
                self.logger.debug("Error details:", exc_info=True)
                stats["failed"] += 1
        
        self.logger.info(f"Batch statistics: {stats}")
        return stats
    
    def find_items_to_process(self):
        """Find items that need processing.
        
        Returns:
            List of data IDs to process
            
        Raises:
            NotImplementedError: Subclasses must implement this method
        """
        raise NotImplementedError("Subclasses must implement find_items_to_process")
    
    def run_pipeline(self, data_ids: Optional[List[str]] = None, batch_size: int = 0, 
                    max_items: int = 0, start_idx: int = 0):
        """Run the pipeline on specified data IDs or find new items to process.
        
        Args:
            data_ids: List of data IDs to process (if None, will find items to process)
            batch_size: Number of items to process in each batch (0 for all at once)
            max_items: Maximum number of items to process (0 for no limit)
            start_idx: Starting index for processing
            
        Returns:
            Dict with processing statistics
        """
        if data_ids is None:
            # Find items that need processing
            self.logger.info("Finding items to process")
            data_ids = self.find_items_to_process()
            
        # Get already processed items
        processed_items = self.get_processed_items()
        unprocessed_ids = [id for id in data_ids if id not in processed_items]
        
        self.logger.info(f"Found {len(unprocessed_ids)} unprocessed items out of {len(data_ids)} total")
        
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
                          help="Dry run mode - don't actually copy files or write to DynamoDB")
    
    @classmethod
    def run_from_command_line(cls):
        """Run the transform from the command line.
        
        This method should be called from the `if __name__ == "__main__"` block
        in subclass implementations.
        """
        # Create argument parser
        parser = argparse.ArgumentParser(description=cls.__doc__)
        
        # Add standard arguments
        cls.add_arguments(parser)
        
        # Add subclass-specific arguments
        cls.add_subclass_arguments(parser)
        
        # Parse arguments
        args = parser.parse_args()
        
        # Create transform instance
        transform = cls.from_args(args)
        
        # Run the pipeline
        transform.run_pipeline(
            batch_size=args.batch_size,
            max_items=args.max_items,
            start_idx=args.start_idx
        )
    
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
        """
        raise NotImplementedError("Subclasses must implement from_args")