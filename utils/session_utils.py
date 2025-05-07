"""
Session utility functions for listing and selecting data sessions.

This module provides functionality to list available sessions in the S3 bucket,
filter them by name and duration, and interactively select sessions for processing.
It also provides utilities to check which sessions have already been processed.
"""

import os
import re
import boto3
import h5py
import tempfile
from datetime import datetime
from typing import Dict, List, Set, Optional, Tuple, Any

from .aws import init_s3_client, init_dynamodb_resource

# S3 bucket and prefixes
BUCKET_NAME = 'conduit-data-dev'
NEW_SESSIONS_PREFIX = 'data-collector/new-sessions/'
TEST_RUNS_PREFIX = 'data-collector/raw-data/test-runs/'
CURATED_H5_PREFIX = 'curated-h5/'

def ms_to_datetime(ms):
    """Convert milliseconds to datetime string"""
    return datetime.fromtimestamp(ms/1000).strftime('%Y-%m-%d %H:%M:%S')

def format_duration(seconds):
    """Format duration in seconds to human-readable format"""
    if seconds < 3600:  # Less than an hour
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        return f"{hours}h {minutes}m {seconds}s"

def list_s3_session_items(s3_client, prefix=NEW_SESSIONS_PREFIX, bucket=BUCKET_NAME):
    """
    List all session items (folders or H5 files) in the prefix
    
    This function handles both folder structures and direct file listings:
    - For raw sessions (data-collector/): Returns folders like 'data-collector/new-sessions/session_xyz/'
    - For curated data (curated-h5/): Returns both folders and H5 files like 'curated-h5/session_xyz.h5'
    """
    paginator = s3_client.get_paginator('list_objects_v2')
    items = set()
    
    # First try to get folders (using delimiter)
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter='/'):
        if 'CommonPrefixes' in page:
            for prefix_obj in page['CommonPrefixes']:
                folder_path = prefix_obj['Prefix']
                items.add(folder_path)
                
    # For prefixes like 'curated-h5/' that might have direct H5 files,
    # we also need to check for files if no folders were found
    if not items or prefix.startswith(CURATED_H5_PREFIX):
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            if 'Contents' in page:
                for obj in page['Contents']:
                    key = obj['Key']
                    # Check if this is a direct H5 file (not in a subfolder)
                    # and not the prefix itself
                    if key.endswith('.h5') and key != prefix and '/' not in key[len(prefix):]:
                        # Add a pseudo-folder path that just points to the file
                        items.add(key)
    
    return items

def list_s3_contents(s3_client, folder, bucket=BUCKET_NAME):
    """List contents of a specific S3 folder"""
    paginator = s3_client.get_paginator('list_objects_v2')
    contents = []

    for page in paginator.paginate(Bucket=bucket, Prefix=folder):
        if 'Contents' in page:
            for obj in page['Contents']:
                contents.append(obj['Key'])

    return contents

def get_session_duration(s3_client, item_path, bucket=BUCKET_NAME):
    """
    Get the duration of a session based on H5 file timestamps or metadata
    
    Args:
        s3_client: Boto3 S3 client
        item_path: Path to session folder or direct H5 file
        bucket: S3 bucket name
        
    Returns:
        Dictionary with session metadata including duration
    """
    # Extract session name from path
    if item_path.endswith('.h5'):
        # Direct H5 file case
        session_name = os.path.basename(item_path).replace('.h5', '')
    else:
        # Folder case
        session_name = item_path.rstrip('/').split('/')[-1]
    
    paginator = s3_client.get_paginator('list_objects_v2')
    
    # Different approaches based on structure
    # 1. Check if this is a direct H5 file in the curated-h5 folder
    # 2. Check if this is a folder with H5 files inside
    # 3. Try to extract timestamps from raw session H5 files
    
    h5_files = []
    is_curated = item_path.startswith(CURATED_H5_PREFIX)
    is_direct_file = item_path.endswith('.h5')
    timestamps = []
    duration_from_metadata = None
    
    # If this is a direct H5 file, just get its metadata
    if is_direct_file:
        h5_files.append(item_path)
        try:
            # Get object metadata to see if we have duration info
            response = s3_client.head_object(Bucket=bucket, Key=item_path)
            metadata = response.get('Metadata', {})
            
            # Check if duration is in the metadata
            if 'duration_sec' in metadata:
                duration_from_metadata = float(metadata['duration_sec'])
            elif 'clean_duration_sec' in metadata:
                duration_from_metadata = float(metadata['clean_duration_sec'])
        except Exception:
            # If we can't get metadata, we'll fall back to timestamp extraction or estimation
            pass
    else:
        # Get all H5 files in this folder
        for page in paginator.paginate(Bucket=bucket, Prefix=item_path):
            if 'Contents' in page:
                for obj in page['Contents']:
                    file_key = obj['Key']
                    file_name = file_key.split('/')[-1]

                    # Only include H5 files that are not in the 'files' subdirectory
                    if file_name.endswith('.h5') and '/files/' not in file_key:
                        h5_files.append(file_key)
                        
                        # For curated files, try to get metadata from the file directly if needed
                        if is_curated and duration_from_metadata is None:
                            try:
                                # Get object metadata to see if we have duration info
                                response = s3_client.head_object(Bucket=bucket, Key=file_key)
                                metadata = response.get('Metadata', {})
                                
                                # Check if duration is in the metadata
                                if 'duration_sec' in metadata:
                                    duration_from_metadata = float(metadata['duration_sec'])
                                elif 'clean_duration_sec' in metadata:
                                    duration_from_metadata = float(metadata['clean_duration_sec'])
                            except Exception:
                                # Continue if we can't get metadata - we'll fall back to timestamp extraction
                                pass
    
    # If it's not a curated file, try to extract timestamps from filenames
    if not is_curated or duration_from_metadata is None:
        timestamp_pattern = re.compile(r'_(\d{13})\.h5$')
        for file_key in h5_files:
            file_name = file_key.split('/')[-1]
            match = timestamp_pattern.search(file_name)
            if match:
                timestamps.append(int(match.group(1)))
    
    # Case 1: We have duration from metadata (curated files)
    if duration_from_metadata is not None:
        duration_sec = duration_from_metadata
        duration_ms = duration_sec * 1000
        
        return {
            'name': session_name,
            'duration': format_duration(duration_sec),
            'duration_ms': duration_ms,
            'duration_minutes': duration_ms / (60 * 1000),
            'h5_count': len(h5_files),
            'h5_files': h5_files,
            'is_curated': True,
            'is_direct_file': is_direct_file
        }
    
    # Case 2: We have timestamps from raw session files
    elif len(timestamps) >= 2:
        timestamps.sort()
        first_ts = timestamps[0]
        last_ts = timestamps[-1]
        duration_ms = last_ts - first_ts

        return {
            'name': session_name,
            'start_time': ms_to_datetime(first_ts),
            'end_time': ms_to_datetime(last_ts),
            'duration': format_duration(duration_ms / 1000),
            'duration_ms': duration_ms,
            'duration_minutes': duration_ms / (60 * 1000),
            'h5_count': len(timestamps),
            'h5_files': h5_files,
            'is_curated': False,
            'is_direct_file': is_direct_file
        }
    
    # Case 3: We couldn't determine duration
    else:
        # Estimate duration based on typical file size to duration ratio if we have files
        if h5_files:
            # For curated files, try to get file size and estimate duration
            try:
                # Get size of the first H5 file
                response = s3_client.head_object(Bucket=bucket, Key=h5_files[0])
                file_size_bytes = response.get('ContentLength', 0)
                
                # Rough estimate: 1MB ~= 10 seconds for raw neural data
                estimated_duration_sec = file_size_bytes / (1024 * 1024) * 10
                duration_ms = estimated_duration_sec * 1000
                
                return {
                    'name': session_name,
                    'duration': format_duration(estimated_duration_sec),
                    'duration_ms': duration_ms,
                    'duration_minutes': duration_ms / (60 * 1000),
                    'h5_count': len(h5_files),
                    'h5_files': h5_files,
                    'is_estimated': True,
                    'is_curated': is_curated,
                    'is_direct_file': is_direct_file
                }
            except Exception:
                pass
        
        # Default case when we can't determine duration
        return {
            'name': session_name,
            'duration': '0s',
            'duration_ms': 0,
            'duration_minutes': 0,
            'h5_count': len(h5_files),
            'h5_files': h5_files,
            'is_curated': is_curated,
            'is_direct_file': is_direct_file
        }

def get_all_sessions(s3_client, prefix=NEW_SESSIONS_PREFIX, bucket=BUCKET_NAME, name_filter=None):
    """Get info for all sessions, with optional name filter"""
    session_items = list_s3_session_items(s3_client, prefix, bucket)
    sessions = {}

    for item in session_items:
        # For H5 files, extract session name from filename
        if item.endswith('.h5'):
            item_name = os.path.basename(item).replace('.h5', '')
        else:
            item_name = item.rstrip('/').split('/')[-1]

        # Apply name filter if provided
        if name_filter and name_filter not in item_name:
            continue

        session_data = get_session_duration(s3_client, item, bucket)
        sessions[item] = session_data

    return sessions

def display_sessions_summary(sessions, min_duration_minutes=0, processed_sessions=None, 
                      include_processed=False, include_skipped=False, skipped_only=False):
    """
    Display summary of sessions with optional processed status indicator.
    
    Args:
        sessions: Dictionary of sessions data
        min_duration_minutes: Minimum session duration in minutes (use 0 to skip duration filtering)
        processed_sessions: Set of session IDs already processed
        include_processed: If True, show already processed sessions; if False, filter them out
        include_skipped: If True, show sessions that were previously skipped
        skipped_only: If True, show only sessions that were previously skipped
        
    Returns:
        List of sorted (folder, data) tuples for sessions meeting criteria
    """
    # Filter by minimum duration (if min_duration_minutes is 0, this will include all sessions)
    filtered_sessions = {
        item_path: data for item_path, data in sessions.items()
        if data.get('duration_minutes', 0) >= min_duration_minutes
    }
    
    # Apply session filtering based on flags
    if processed_sessions is not None:
        # Get processed and skipped sessions from DynamoDB
        if skipped_only:
            # Only include sessions that were previously skipped
            filtered_sessions = {
                item_path: data for item_path, data in filtered_sessions.items()
                if data['name'] in processed_sessions
            }
        elif include_processed and include_skipped:
            # Include all sessions (no filtering)
            pass
        elif include_processed:
            # Only filter out skipped sessions (if we had that information separately)
            # Since we don't, include all sessions
            pass
        elif include_skipped:
            # Include skipped sessions, filter out successfully processed ones
            # Would require separate lists of success/skipped, which we don't have
            # For now, no filtering is applied
            pass
        else:
            # Default: filter out all processed sessions (both success and skipped)
            filtered_sessions = {
                item_path: data for item_path, data in filtered_sessions.items()
                if data['name'] not in processed_sessions
            }

    # Sort sessions by duration (longest first)
    sorted_sessions = sorted(filtered_sessions.items(), key=lambda x: x[1]['duration_ms'], reverse=True)

    # Display results
    has_curated = any(data.get('is_curated', False) for _, data in sorted_sessions)
    has_processed = False  # Flag to indicate if we have any processed sessions
    
    # Define table headers
    header_format = f"{'Session Name':<35} {'Type':<12} {'Start Time':<20} {'End Time/Info':<20} {'Duration':<15} {'Files':<5}"
    if processed_sessions is not None:
        header_format += " {'Status':<10}"
        has_processed = any(data['name'] in processed_sessions for _, data in sorted_sessions)
    
    print(header_format)
    print("-" * (110 + (12 if processed_sessions is not None else 0)))

    unprocessed_count = 0
    for item_path, data in sorted_sessions:
        session_type = "Curated" if data.get('is_curated', False) else "Raw"
        is_file = item_path.endswith('.h5')
        item_indicator = "[File]" if is_file else "[Folder]"
        
        # Build basic line without status
        if 'start_time' in data:
            # For raw sessions with start/end times
            line = f"{data['name']:<35} {session_type+item_indicator:<12} {data['start_time']:<20} {data['end_time']:<20} {data['duration']:<15} {data['h5_count']:<5}"
        else:
            # For curated sessions or those without timestamps
            estimated = "(estimated)" if data.get('is_estimated', False) else ""
            info_field = f"{estimated}"
            line = f"{data['name']:<35} {session_type+item_indicator:<12} {'N/A':<20} {info_field:<20} {data['duration']:<15} {data['h5_count']:<5}"
            
        # Add processed status if we're tracking it
        if processed_sessions is not None:
            is_processed = data['name'] in processed_sessions
            status = "PROCESSED" if is_processed else "NEW"
            line += f" {status:<10}"
            if not is_processed:
                unprocessed_count += 1
        else:
            unprocessed_count += 1
            
        print(line)

    # Print summary with additional processed info if applicable
    if processed_sessions is not None and has_processed:
        print(f"\nTotal sessions matching criteria: {len(sorted_sessions)}")
        print(f"  - Already processed: {len(sorted_sessions) - unprocessed_count}")
        print(f"  - New (unprocessed): {unprocessed_count}")
        if not include_processed:
            print(f"NOTE: Only showing NEW sessions (use --include-processed to see all)")
    else:
        if min_duration_minutes > 0:
            print(f"\nTotal sessions with duration > {min_duration_minutes} minutes: {len(sorted_sessions)}")
        else:
            print(f"\nTotal sessions: {len(sorted_sessions)}")

    return sorted_sessions

def find_main_h5_file(s3_client, session_path, session_name, bucket=BUCKET_NAME):
    """Find the main H5 file in the root directory of a session"""
    # If the session_path is already a direct H5 file, just return it
    if session_path.endswith('.h5'):
        return session_path
        
    paginator = s3_client.get_paginator('list_objects_v2')

    # Look for session_name.h5 pattern
    main_h5_file = None

    for page in paginator.paginate(Bucket=bucket, Prefix=session_path):
        if 'Contents' in page:
            for obj in page['Contents']:
                file_key = obj['Key']
                file_name = file_key.split('/')[-1]

                # Skip files in subdirectories
                if '/' in file_name:
                    continue

                # Check if this is the main h5 file (not in the 'files' subdirectory)
                # It should match the session name or be named with the session name
                if file_name.endswith('.h5') and ('files/' not in file_key):
                    # If the filename exactly matches session_name.h5, it's definitely the main file
                    if file_name == f"{session_name}.h5":
                        return file_key

                    # Otherwise, keep this as a candidate
                    main_h5_file = file_key

    return main_h5_file

def download_h5_file(s3_client, s3_path, local_path, bucket=BUCKET_NAME):
    """Download an H5 file from S3 to a local path"""
    try:
        print(f"Downloading {s3_path} to {local_path}...")
        s3_client.download_file(bucket, s3_path, local_path)
        return True
    except Exception as e:
        print(f"Error downloading {s3_path}: {e}")
        return False

def print_h5_structure(file_path, max_array_items=5):
    """Print the structure of an H5 file, including dataset shapes and sample data"""
    def h5_explorer(name, obj):
        if isinstance(obj, h5py.Dataset):
            shape_str = str(obj.shape)
            dtype_str = str(obj.dtype)

            # Handle different data types
            if obj.dtype.kind in ['i', 'f', 'u']:  # Integer, float, or unsigned int
                if len(obj.shape) == 0:  # Scalar value
                    value = f"Value: {obj[()]}"
                elif obj.size > 0:  # Non-empty array
                    if len(obj.shape) == 1 and obj.shape[0] <= max_array_items:
                        value = f"Values: {obj[:]}"
                    else:
                        # For larger arrays, just show a few items
                        flat_array = obj[(0,) * (len(obj.shape) - 1)].flatten()
                        if len(flat_array) > 0:
                            sample = flat_array[:min(max_array_items, len(flat_array))]
                            value = f"Sample: {sample}..."
                        else:
                            value = "Empty array"
                else:
                    value = "Empty array"
            elif obj.dtype.kind == 'S' or obj.dtype.kind == 'O':  # String or object
                if len(obj.shape) == 0:  # Scalar string
                    try:
                        value = f"Value: {obj[()].decode('utf-8') if isinstance(obj[()], bytes) else obj[()]}"
                    except:
                        value = f"Value: {obj[()]}"
                elif obj.size > 0:  # Non-empty array
                    try:
                        if len(obj.shape) == 1 and obj.shape[0] <= max_array_items:
                            string_values = [item.decode('utf-8') if isinstance(item, bytes) else item for item in obj[:]]
                            value = f"Values: {string_values}"
                        else:
                            # For larger arrays, just show a few items
                            sample_items = obj.flatten()[:min(max_array_items, obj.size)]
                            string_values = [item.decode('utf-8') if isinstance(item, bytes) else item for item in sample_items]
                            value = f"Sample: {string_values}..."
                    except:
                        value = "String array (unable to decode)"
                else:
                    value = "Empty array"
            else:
                value = "Complex data type"

            print(f"{name}: Dataset, Shape: {shape_str}, Type: {dtype_str}, {value}")
        elif isinstance(obj, h5py.Group):
            print(f"{name}: Group")

    try:
        with h5py.File(file_path, 'r') as f:
            print(f"\nH5 File Structure for: {os.path.basename(file_path)}")
            print("-" * 80)
            f.visititems(h5_explorer)

            # List root level attributes
            print("\nRoot Level Attributes:")
            for attr in f.attrs:
                try:
                    value = f.attrs[attr]
                    if isinstance(value, (list, tuple)) and len(value) > max_array_items:
                        value = f"{value[:max_array_items]}..."
                    print(f"  {attr}: {value}")
                except:
                    print(f"  {attr}: <Unable to read attribute>")
    except Exception as e:
        print(f"Error reading H5 file: {e}")

def inspect_session_h5(s3_client, session_path, session_name, bucket=BUCKET_NAME):
    """Inspect the main H5 file in a session directory"""
    # Find the main H5 file for this session
    main_h5_file = find_main_h5_file(s3_client, session_path, session_name, bucket)

    if not main_h5_file:
        print(f"No main H5 file found for session: {session_name}")
        return None

    # Create a temporary file to download to
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as temp_file:
        temp_path = temp_file.name

    try:
        # Download the file
        if download_h5_file(s3_client, main_h5_file, temp_path, bucket):
            # Analyze the file structure
            print_h5_structure(temp_path)
            # Return the S3 key of the main file
            return main_h5_file
        else:
            return None
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

def get_processed_sessions(transform_id, dynamodb_table='conduit-pipeline-metadata', include_skipped=True, skipped_only=False):
    """
    Get a set of session IDs that have already been processed by the specified transform.
    
    Args:
        transform_id: ID of the transform to check
        dynamodb_table: Name of the DynamoDB table with transform records
        include_skipped: Whether to include items with 'skipped' status as processed
        skipped_only: If True, return only skipped sessions, ignoring success sessions
        
    Returns:
        Set of session IDs that have been processed
    """
    processed_sessions = set()
    
    try:
        # Initialize DynamoDB
        dynamodb = init_dynamodb_resource()
        table = dynamodb.Table(dynamodb_table)
        
        # Use the TransformIndex to get processed transforms for this transform_id
        from boto3.dynamodb.conditions import Key, Attr
        
        # Start with an empty LastEvaluatedKey
        last_key = None
        
        # Build the filter expression based on flags
        if skipped_only:
            # Only include 'skipped' status
            filter_expr = Attr('status').eq('skipped')
        elif include_skipped:
            # Include both 'success' and 'skipped' status
            filter_expr = Attr('status').is_in(['success', 'skipped'])
        else:
            # Only include 'success' status
            filter_expr = Attr('status').eq('success')
        
        while True:
            # Prepare query params
            query_params = {
                'IndexName': 'TransformIndex',
                'KeyConditionExpression': Key('transform_id').eq(transform_id),
                'FilterExpression': filter_expr,
                'ProjectionExpression': 'data_id'
            }
            
            # Add LastEvaluatedKey for pagination if we have one
            if last_key:
                query_params['ExclusiveStartKey'] = last_key
                
            # Execute the query
            response = table.query(**query_params)
            
            # Add all data_ids to our set
            for item in response.get('Items', []):
                processed_sessions.add(item['data_id'])
                
            # Check if we need to paginate
            if 'LastEvaluatedKey' in response:
                last_key = response['LastEvaluatedKey']
            else:
                break
                
        # Create message based on what we're including
        if skipped_only:
            status_msg = "previously skipped"
        elif include_skipped:
            status_msg = "successfully processed or skipped"
        else:
            status_msg = "successfully processed"
        print(f"Found {len(processed_sessions)} sessions {status_msg} by {transform_id}")
        return processed_sessions
        
    except Exception as e:
        print(f"Error getting processed sessions: {e}")
        return set()

def select_sessions_interactive(sessions, min_duration_minutes=0, processed_sessions=None, 
                            include_processed=False, include_skipped=False, skipped_only=False):
    """
    Interactive session selection using the same flow as in inspect_new_sessions.py
    
    Args:
        sessions: Dictionary of session data
        min_duration_minutes: Minimum session duration in minutes
        processed_sessions: Set of session IDs that have already been processed
        include_processed: Whether to include already processed sessions in display and selection
        include_skipped: Whether to include previously skipped sessions
        skipped_only: Whether to show only previously skipped sessions
    """
    # Display summary
    print("Available sessions matching criteria:")
    sorted_sessions = display_sessions_summary(sessions, min_duration_minutes, processed_sessions,
                                     include_processed, include_skipped, skipped_only)
    print("\n")
    
    selected_sessions = []
    
    # Always prompt for selection, even with only one session
    if len(sorted_sessions) >= 1:
        if len(sorted_sessions) == 1:
            print("\nOne session found. Please confirm:")
        else:
            print("\nMultiple sessions found. Options:")
        for i, (folder, data) in enumerate(sorted_sessions, 1):
            session_id = data['name']
            processed_marker = " [PROCESSED]" if processed_sessions and session_id in processed_sessions else ""
            print(f"  {i}: {session_id} ({data['duration']}){processed_marker}")
        print("  all: Process all sessions in the list")
        print("  quit: Exit without processing")

        choice = input("\nEnter choice (number, 'all', or 'quit'): ").strip().lower()

        if choice == 'quit':
            return []
        elif choice == 'all':
            # Simply return all sessions currently shown in the list
            print(f"\nProcessing all {len(sorted_sessions)} sessions in the list")
            return sorted_sessions
        elif choice.isdigit() and 1 <= int(choice) <= len(sorted_sessions):
            idx = int(choice) - 1
            folder, data = sorted_sessions[idx]
            print(f"\nSelected session: {data['name']}")
            selected_sessions.append((folder, data))
        else:
            print("Invalid choice. Exiting.")
            return []
    else:
        print("No sessions found matching criteria.")
        
    return selected_sessions