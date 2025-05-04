#!/usr/bin/env python
import os
import re
import sys
import argparse
import tempfile
import json
import h5py
import numpy as np
from datetime import datetime
import boto3
from botocore.exceptions import ClientError

# S3 bucket and prefixes
BUCKET_NAME = 'conduit-data-dev'
NEW_SESSIONS_PREFIX = 'data-collector/new-sessions/'
TEST_RUNS_PREFIX = 'data-collector/raw-data/test-runs/'
PREFIX = NEW_SESSIONS_PREFIX  # Default prefix

# Function to load AWS credentials from env.secrets
def load_aws_credentials():
    secrets = {}
    with open('../.env.secrets', 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                key, value = line.split('=', 1)
                secrets[key] = value
    return secrets

# Initialize S3 client
def init_s3_client():
    secrets = load_aws_credentials()
    return boto3.client(
        's3',
        aws_access_key_id=secrets.get('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=secrets.get('AWS_SECRET_ACCESS_KEY'),
        region_name=secrets.get('AWS_DEFAULT_REGION', 'us-east-1')
    )

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

def list_s3_folders(s3_client):
    """List all folders in the prefix"""
    paginator = s3_client.get_paginator('list_objects_v2')
    folders = set()

    for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix=PREFIX, Delimiter='/'):
        if 'CommonPrefixes' in page:
            for prefix_obj in page['CommonPrefixes']:
                folder_path = prefix_obj['Prefix']
                folders.add(folder_path)

    return folders

def list_s3_contents(s3_client, folder):
    """List contents of a specific S3 folder"""
    paginator = s3_client.get_paginator('list_objects_v2')
    contents = []

    for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix=folder):
        if 'Contents' in page:
            for obj in page['Contents']:
                contents.append(obj['Key'])

    return contents

def get_session_duration(s3_client, folder):
    """Get the duration of a session based on H5 file timestamps"""
    folder_name = folder.rstrip('/').split('/')[-1]
    paginator = s3_client.get_paginator('list_objects_v2')
    timestamp_pattern = re.compile(r'_(\d{13})\.h5$')
    timestamps = []
    h5_files = []

    for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix=folder):
        if 'Contents' in page:
            for obj in page['Contents']:
                file_key = obj['Key']
                file_name = file_key.split('/')[-1]

                if file_name.endswith('.h5'):
                    h5_files.append(file_key)
                    match = timestamp_pattern.search(file_name)
                    if match:
                        timestamps.append(int(match.group(1)))

    if len(timestamps) < 2:
        return {
            'name': folder_name,
            'duration': '0s',
            'duration_ms': 0,
            'duration_minutes': 0,
            'h5_count': len(timestamps),
            'h5_files': h5_files
        }

    timestamps.sort()
    first_ts = timestamps[0]
    last_ts = timestamps[-1]
    duration_ms = last_ts - first_ts

    return {
        'name': folder_name,
        'start_time': ms_to_datetime(first_ts),
        'end_time': ms_to_datetime(last_ts),
        'duration': format_duration(duration_ms / 1000),
        'duration_ms': duration_ms,
        'duration_minutes': duration_ms / (60 * 1000),
        'h5_count': len(timestamps),
        'h5_files': h5_files
    }

def get_all_sessions(s3_client, name_filter=None):
    """Get info for all sessions, with optional name filter"""
    folders = list_s3_folders(s3_client)
    sessions = {}

    for folder in folders:
        folder_name = folder.rstrip('/').split('/')[-1]

        # Apply name filter if provided
        if name_filter and name_filter not in folder_name:
            continue

        session_data = get_session_duration(s3_client, folder)
        sessions[folder] = session_data

    return sessions

def display_sessions_summary(sessions, min_duration_minutes=0):
    """Display summary of sessions"""
    # Filter by minimum duration
    filtered_sessions = {
        folder: data for folder, data in sessions.items()
        if data['duration_minutes'] >= min_duration_minutes
    }

    # Sort sessions by duration (longest first)
    sorted_sessions = sorted(filtered_sessions.items(), key=lambda x: x[1]['duration_ms'], reverse=True)

    # Display results
    print(f"{'Session Name':<35} {'Start Time':<20} {'End Time':<20} {'Duration':<15} {'Files':<5}")
    print("-" * 95)

    for folder, data in sorted_sessions:
        if 'start_time' in data:
            print(f"{data['name']:<35} {data['start_time']:<20} {data['end_time']:<20} {data['duration']:<15} {data['h5_count']:<5}")
        else:
            print(f"{data['name']:<35} {'N/A':<20} {'N/A':<20} {data['duration']:<15} {data['h5_count']:<5}")

    print(f"\nTotal sessions with duration > {min_duration_minutes} minutes: {len(sorted_sessions)}")

    return sorted_sessions

def download_h5_file(s3_client, s3_path, local_path):
    """Download an H5 file from S3 to a local path"""
    try:
        print(f"Downloading {s3_path} to {local_path}...")
        s3_client.download_file(BUCKET_NAME, s3_path, local_path)
        return True
    except ClientError as e:
        print(f"Error downloading {s3_path}: {e}")
        return False

def find_main_h5_file(s3_client, session_path, session_name):
    """Find the main H5 file in the root directory of a session"""
    paginator = s3_client.get_paginator('list_objects_v2')

    # Look for session_name.h5 pattern
    main_h5_file = None

    for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix=session_path):
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
                elif np.prod(obj.shape) > 0:  # Non-empty array
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
                elif np.prod(obj.shape) > 0:  # Non-empty array
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
                    if isinstance(value, np.ndarray) and value.size > max_array_items:
                        value = f"{value[:max_array_items]}..."
                    print(f"  {attr}: {value}")
                except:
                    print(f"  {attr}: <Unable to read attribute>")
    except Exception as e:
        print(f"Error reading H5 file: {e}")

def inspect_session_h5(s3_client, session_path, session_name):
    """Inspect the main H5 file in a session directory"""
    # Find the main H5 file for this session
    main_h5_file = find_main_h5_file(s3_client, session_path, session_name)

    if not main_h5_file:
        print(f"No main H5 file found for session: {session_name}")
        return None

    # Create a temporary file to download to
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as temp_file:
        temp_path = temp_file.name

    try:
        # Download the file
        if download_h5_file(s3_client, main_h5_file, temp_path):
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

def main():
    parser = argparse.ArgumentParser(description='Inspect H5 files in S3 new-sessions directory')
    parser.add_argument('--min-duration', type=int, default=10,
                      help='Minimum session duration in minutes (default: 10)')
    parser.add_argument('--filter', type=str,
                      help='Filter session names containing this string')
    parser.add_argument('--session', type=str,
                      help='Inspect a specific session by name')
    parser.add_argument('--list-only', action='store_true',
                      help='Only list available sessions without inspecting')
    parser.add_argument('--test', action='store_true',
                      help='Use test-runs directory instead of new-sessions')

    args = parser.parse_args()

    # Select the appropriate prefix based on the --test flag
    global PREFIX
    PREFIX = TEST_RUNS_PREFIX if args.test else NEW_SESSIONS_PREFIX
    print(f"Using S3 path: s3://{BUCKET_NAME}/{PREFIX}")

    # Initialize S3 client
    s3_client = init_s3_client()

    # Get all sessions data with optional filter
    sessions = get_all_sessions(s3_client, args.filter)

    # Display summary
    print("Available sessions matching criteria:")
    sorted_sessions = display_sessions_summary(sessions, args.min_duration)
    print("\n")

    # If list-only is specified, exit after showing the summary
    if args.list_only:
        sys.exit(0)

    # If a specific session is provided, only inspect that one
    if args.session:
        found = False
        for folder, data in sorted_sessions:
            if args.session in data['name']:
                print(f"Inspecting session: {data['name']}")
                inspect_session_h5(s3_client, folder, data['name'])
                found = True
                break

        if not found:
            print(f"No session found matching: {args.session}")
    else:
        # Auto-select if only one session
        if len(sorted_sessions) == 1:
            folder, data = sorted_sessions[0]
            print(f"\nInspecting single matching session: {data['name']}")
            inspect_session_h5(s3_client, folder, data['name'])
        # Prompt for which session to inspect if multiple
        elif len(sorted_sessions) > 1:
            print("\nMultiple sessions found. Options:")
            for i, (folder, data) in enumerate(sorted_sessions, 1):
                print(f"  {i}: {data['name']} ({data['duration']})")
            print("  all: Inspect all matching sessions")
            print("  quit: Exit without inspecting")

            choice = input("\nEnter choice (number, 'all', or 'quit'): ").strip().lower()

            if choice == 'quit':
                sys.exit(0)
            elif choice == 'all':
                for folder, data in sorted_sessions:
                    print(f"\n{'='*80}\nInspecting session: {data['name']}\n{'='*80}")
                    inspect_session_h5(s3_client, folder, data['name'])
            elif choice.isdigit() and 1 <= int(choice) <= len(sorted_sessions):
                idx = int(choice) - 1
                folder, data = sorted_sessions[idx]
                print(f"\nInspecting session: {data['name']}")
                inspect_session_h5(s3_client, folder, data['name'])
            else:
                print("Invalid choice. Exiting.")
        else:
            print("No sessions found matching criteria.")

if __name__ == "__main__":
    main()
