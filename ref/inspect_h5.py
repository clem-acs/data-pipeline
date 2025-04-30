#!/usr/bin/env python
import os
import boto3
import tempfile
import h5py
import re
import json
from datetime import datetime

# S3 bucket details
BUCKET_NAME = "conduit-data-dev"
PREFIX = "data-collector/new-sessions/"

def init_s3_client():
    """Initialize S3 client with AWS credentials from environment"""
    return boto3.client(
        's3',
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        region_name=os.environ.get("AWS_REGION", "us-east-1")
    )

def get_session_files(s3_client, session_path):
    """Get all files in a session, both the main file and files in /files directory"""
    paginator = s3_client.get_paginator('list_objects_v2')
    main_h5_files = []
    sub_h5_files = []

    for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix=session_path):
        if 'Contents' in page:
            for obj in page['Contents']:
                file_key = obj['Key']
                file_name = file_key.split('/')[-1]

                # Separate files in the 'files' subdirectory
                if '/files/' in file_key and file_name.endswith('.h5'):
                    sub_h5_files.append((file_key, obj['Size']))
                # Get direct files in the session folder that are H5 files
                elif file_name.endswith('.h5'):
                    main_h5_files.append((file_key, obj['Size']))

    # Sort files by name
    if main_h5_files:
        main_h5_files.sort(key=lambda x: x[0])
    if sub_h5_files:
        sub_h5_files.sort(key=lambda x: x[0])

    return main_h5_files, sub_h5_files

def find_main_h5_file(s3_client, session_path):
    """Find the main H5 file in the root directory of a session (not in /files)"""
    main_h5_files, _ = get_session_files(s3_client, session_path)

    if main_h5_files:
        # Return the key of the first file
        return main_h5_files[0][0]

    return None

def download_h5_file(s3_client, s3_path, local_path, partial=False, max_size_mb=10):
    """Download an H5 file from S3 to a local path

    Args:
        s3_client: The boto3 S3 client
        s3_path: The S3 key to download
        local_path: The local path to save to
        partial: If True, only download the file header (first part)
        max_size_mb: Max size in MB to download if partial=True
    """
    try:
        if not partial:
            # Download the entire file
            s3_client.download_file(BUCKET_NAME, s3_path, local_path)
            return True
        else:
            # Get the file metadata first
            response = s3_client.head_object(Bucket=BUCKET_NAME, Key=s3_path)
            total_size = response['ContentLength']

            # Determine how much to download (in bytes)
            max_size_bytes = max_size_mb * 1024 * 1024
            download_size = min(total_size, max_size_bytes)

            print(f"File size: {total_size / (1024*1024):.2f} MB, downloading first {download_size / (1024*1024):.2f} MB")

            # Download partial file
            response = s3_client.get_object(
                Bucket=BUCKET_NAME,
                Key=s3_path,
                Range=f"bytes=0-{download_size-1}"
            )

            # Write the partial content to the local file
            with open(local_path, 'wb') as f:
                f.write(response['Body'].read())

            return True
    except Exception as e:
        print(f"Error downloading {s3_path}: {str(e)}")
        return False

def print_h5_structure(file_path):
    """Print structure of an H5 file with basic information"""
    try:
        with h5py.File(file_path, 'r') as f:
            # Print root attributes
            print(f"\nRoot level attributes:")
            for attr_name, attr_value in f.attrs.items():
                print(f"  {attr_name}: {attr_value}")

            # Print top-level groups and datasets
            print("\nTop-level groups and datasets:")
            for name in f:
                if isinstance(f[name], h5py.Group):
                    print(f"  Group: {name}")

                    # Check for devices group specifically and show its subgroups
                    if name == "devices" and "devices" in f:
                        print("\n  Devices subgroups:")
                        devices_group = f["devices"]

                        # List all device subgroups
                        for device_name in devices_group:
                            print(f"    Device: {device_name}")

                            # Show attributes of this device if any
                            device = devices_group[device_name]
                            if len(device.attrs) > 0:
                                print(f"      Attributes:")
                                for attr_name, attr_value in device.attrs.items():
                                    print(f"        {attr_name}: {attr_value}")
                else:
                    print(f"  Dataset: {name} (shape: {f[name].shape}, dtype: {f[name].dtype})")
    except Exception as e:
        print(f"Error reading H5 file: {e}")

def extract_timestamp_from_filename(filename):
    """Extract timestamp from filename, assuming it ends with unix timestamp before .h5"""
    try:
        # Extract the unix timestamp (milliseconds) from filename
        timestamp_str = filename.split('_')[-1].split('.')[0]
        return int(timestamp_str)
    except (IndexError, ValueError):
        return None

def calculate_data_rate(total_size_bytes, duration_ms):
    """Calculate data rate in KB/s"""
    if duration_ms <= 0:
        return 0

    # Convert bytes to KB and ms to seconds
    size_kb = total_size_bytes / 1024
    duration_sec = duration_ms / 1000

    return size_kb / duration_sec

def find_task_events_file(s3_client, session_path):
    """Find the task_events.json file in a session directory"""
    paginator = s3_client.get_paginator('list_objects_v2')

    for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix=session_path):
        if 'Contents' in page:
            for obj in page['Contents']:
                file_key = obj['Key']
                if file_key.endswith('task_events.json'):
                    return file_key

    return None

def get_client_timestamps_from_events(s3_client, task_events_key):
    """Extract client_timestamp values from task_events.json file"""
    try:
        response = s3_client.get_object(Bucket=BUCKET_NAME, Key=task_events_key)
        content = response['Body'].read().decode('utf-8')
        events = json.loads(content)

        timestamps = []
        for event in events:
            if 'client_timestamp' in event:
                timestamps.append(event['client_timestamp'])

        if timestamps:
            timestamps.sort()
            return timestamps[0], timestamps[-1]

        return None, None
    except Exception as e:
        print(f"Error parsing task events file: {e}")
        return None, None

def classify_data_rate(rate_kb_per_sec):
    """Classify the data rate into device categories"""
    if rate_kb_per_sec < 100:
        return "neither"
    elif rate_kb_per_sec < 200:
        return "fnirs only"
    else:
        return "both (fnirs+eeg)"

def main():
    print("Initializing S3 client...")
    s3_client = init_s3_client()

    print(f"Looking for H5 files in s3://{BUCKET_NAME}/{PREFIX}")

    # Create a temporary file for downloading
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as temp_file:
        temp_path = temp_file.name

    try:
        # List all session directories
        paginator = s3_client.get_paginator('list_objects_v2')
        prefixes = []

        for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix=PREFIX, Delimiter='/'):
            if 'CommonPrefixes' in page:
                for prefix_obj in page['CommonPrefixes']:
                    prefixes.append(prefix_obj['Prefix'])

        print(f"Found {len(prefixes)} session directories")

        # Process each session directory
        for session_prefix in prefixes:
            session_name = session_prefix.strip('/').split('/')[-1]
            print(f"\n======= Session: {session_name} =======")

            # First check for task_events.json
            task_events_key = find_task_events_file(s3_client, session_prefix)
            if not task_events_key:
                print("No task_events.json found, skipping session")
                continue

            print(f"Found task_events.json: {task_events_key}")

            # Get client timestamps from task events
            first_timestamp, last_timestamp = get_client_timestamps_from_events(s3_client, task_events_key)
            if first_timestamp is None or last_timestamp is None:
                print("Could not extract client timestamps from task_events.json")
                clean_duration_ms = None
            else:
                clean_duration_ms = last_timestamp - first_timestamp
                clean_duration_min = clean_duration_ms / (1000 * 60)
                print(f"Clean duration from task events: {clean_duration_min:.2f} minutes ({clean_duration_ms/1000:.1f} seconds)")

            # Get all session files
            main_h5_files, sub_h5_files = get_session_files(s3_client, session_prefix)

            if main_h5_files:
                main_file_key, main_file_size = main_h5_files[0]
                filename = main_file_key.split('/')[-1]
                print(f"Found main H5 file: {filename} ({main_file_size/(1024*1024):.2f} MB)")

                # Calculate session duration and data rate if we have files with timestamps
                if sub_h5_files:
                    # Extract timestamps from filenames
                    timestamps = []
                    total_size = main_file_size  # Start with main file size

                    for file_key, file_size in sub_h5_files:
                        filename = file_key.split('/')[-1]
                        timestamp = extract_timestamp_from_filename(filename)
                        if timestamp:
                            timestamps.append(timestamp)
                        total_size += file_size

                    if timestamps:
                        # Calculate duration from earliest to latest timestamp
                        earliest = min(timestamps)
                        latest = max(timestamps)
                        duration_ms = latest - earliest
                        duration_min = duration_ms / (1000 * 60)

                        print(f"File-based session duration: {duration_min:.2f} minutes ({duration_ms/1000:.1f} seconds)")
                        print(f"Total data size: {total_size/(1024*1024):.2f} MB")

                        # Calculate and classify data rates - use clean duration if available
                        if clean_duration_ms and clean_duration_ms > 0:
                            clean_data_rate = calculate_data_rate(total_size, clean_duration_ms)
                            device_classification = classify_data_rate(clean_data_rate)
                            print(f"Clean data rate: {clean_data_rate:.2f} KB/s - Classification: {device_classification}")

                        # Always show file-based data rate as well
                        file_data_rate = calculate_data_rate(total_size, duration_ms)
                        file_device_classification = classify_data_rate(file_data_rate)
                        print(f"File-based data rate: {file_data_rate:.2f} KB/s - Classification: {file_device_classification}")

                        # For small files (< 10MB), download and inspect the main file
                        if main_file_size < 10 * 1024 * 1024:
                            print(f"Main file size: {main_file_size/(1024*1024):.2f} MB, downloading for inspection")
                            if download_h5_file(s3_client, main_file_key, temp_path, partial=False):
                                print_h5_structure(temp_path)
                        else:
                            print(f"Main file size: {main_file_size/(1024*1024):.2f} MB - too large for detailed inspection")
                    else:
                        print("No valid timestamps found in files")
                else:
                    # No timestamped files found, just inspect the main file if it's small enough
                    if main_file_size < 10 * 1024 * 1024:
                        if download_h5_file(s3_client, main_file_key, temp_path, partial=False):
                            print_h5_structure(temp_path)
                    else:
                        print(f"File size: {main_file_size/(1024*1024):.2f} MB - too large for complete download")
            else:
                print("No main H5 file found in session root directory")

    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    main()
