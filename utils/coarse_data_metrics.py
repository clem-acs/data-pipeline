import boto3
import json
import os
from datetime import datetime

# AWS credentials and configuration
AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')
AWS_REGION = os.environ.get('AWS_DEFAULT_REGION', 'us-east-1')
S3_BUCKET = 'conduit-data-dev'
NEW_SESSIONS_PREFIX = 'data-collector/'

# Global state variables
total_hours = 0
last_updated = None
session_details = []

def classify_data_rate(rate_kb_per_sec):
    """Classify the data rate into device categories"""
    if rate_kb_per_sec < 70:
        return "neither"
    elif rate_kb_per_sec < 200:
        return "fnirs only"
    else:
        return "both (fnirs+eeg)"

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

    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=session_path):
        if 'Contents' in page:
            for obj in page['Contents']:
                file_key = obj['Key']
                if file_key.endswith('task_events.json'):
                    return file_key

    return None

def find_element_events_file(s3_client, session_path):
    """Find the element_events.json file in a session directory"""
    paginator = s3_client.get_paginator('list_objects_v2')

    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=session_path):
        if 'Contents' in page:
            for obj in page['Contents']:
                file_key = obj['Key']
                if file_key.endswith('element_events.json'):
                    return file_key

    return None

def get_client_timestamps_from_events(s3_client, events_key):
    """Extract client_timestamp values from element_events.json or task_events.json file.

    Now using the first event as start time and the last event as end time.
    """
    try:
        response = s3_client.get_object(Bucket=S3_BUCKET, Key=events_key)
        content = response['Body'].read().decode('utf-8')
        events = json.loads(content)

        # Check if we have any events
        if not events:
            print(f"No events found in events file")
            return None, None

        # Get first and last event timestamps
        start_time = None
        end_time = None

        if len(events) > 0 and 'client_timestamp' in events[0]:
            start_time = events[0]['client_timestamp']

        if len(events) > 0 and 'client_timestamp' in events[-1]:
            end_time = events[-1]['client_timestamp']

        if start_time and end_time:
            return start_time, end_time

        print("Could not find required timestamps in events")
        return None, None
    except Exception as e:
        print(f"Error parsing events file: {e}")
        return None, None


def get_session_files_and_size(s3_client, session_path):
    """Get all files in a session and their total size"""
    paginator = s3_client.get_paginator('list_objects_v2')
    main_h5_files = []
    sub_h5_files = []

    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=session_path):
        if 'Contents' in page:
            for obj in page['Contents']:
                file_key = obj['Key']
                file_name = file_key.split('/')[-1]
                file_size = obj['Size']

                # Get direct files in the session folder that are H5 files
                # Ignore files in the 'files' subdirectory
                if file_name.endswith('.h5') and '/files/' not in file_key:
                    main_h5_files.append((file_key, file_size))

    # Calculate total size of files - only use main H5 files
    total_size = sum(size for _, size in main_h5_files)

    return main_h5_files, sub_h5_files, total_size

async def calculate_total_hours() -> float:
    """Calculate total hours based on first and last timestamped file in each session,
    including task event analysis."""
    global total_hours, last_updated, session_details

    # Create a backup of previous session details to detect changes
    previous_session_count = len(session_details)

    if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY or not S3_BUCKET:
        print("AWS credentials or bucket not configured")
        return 0

    print(f"Starting calculation with bucket: {S3_BUCKET}, prefix: {NEW_SESSIONS_PREFIX}")
    try:
        s3 = boto3.client(
            's3',
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION
        )

        # List all session directories
        paginator = s3.get_paginator('list_objects_v2')
        prefixes = []

        for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=NEW_SESSIONS_PREFIX, Delimiter='/'):
            if 'CommonPrefixes' in page:
                for prefix_obj in page['CommonPrefixes']:
                    prefixes.append(prefix_obj['Prefix'])

        print(f"Found {len(prefixes)} session directories")

        # Reset session details
        session_details = []

        # Calculate durations for each session
        session_durations = []
        valid_sessions = 0

        # Process each session directory
        for session_prefix in prefixes:
            session_id = session_prefix.strip('/').split('/')[-1]
            print(f"\n======= Session: {session_id} =======")

            # First check for task_events.json
            task_events_key = find_task_events_file(s3, session_prefix)
            if not task_events_key:
                print(f"No task_events.json found in session {session_id}, setting clean duration to 0")
                clean_duration_ms = 0
                clean_duration_min = 0
            else:
                print(f"Found task_events.json: {task_events_key}")

                # Get client timestamps from task events
                first_timestamp, last_timestamp = get_client_timestamps_from_events(s3, task_events_key)
                if first_timestamp is None or last_timestamp is None:
                    print(f"Could not extract client timestamps from task_events for {session_id}, setting clean duration to 0")
                    clean_duration_ms = 0
                    clean_duration_min = 0
                else:
                    clean_duration_ms = last_timestamp - first_timestamp
                    clean_duration_min = clean_duration_ms / (1000 * 60)
                    print(f"Clean duration from task events: {clean_duration_min:.2f} minutes ({clean_duration_ms/1000:.1f} seconds)")

            # Get all session files
            main_h5_files, sub_h5_files, total_size = get_session_files_and_size(s3, session_prefix)

            if not main_h5_files:
                print(f"No main H5 file found in session {session_id}")
                # Don't continue here, allow session to be processed with 0 size files

            # Extract timestamps from filenames for file-based duration
            timestamps = []
            for file_key, _ in sub_h5_files:
                filename = file_key.split('/')[-1]
                try:
                    timestamp_str = filename.split('_')[-1].split('.')[0]
                    timestamp = int(timestamp_str)
                    timestamps.append(timestamp)
                except (IndexError, ValueError) as e:
                    print(f"Error parsing timestamp from {filename}: {e}")

            if not timestamps:
                print(f"No valid timestamps found in files for session {session_id}, setting file duration to 0")
                file_duration_ms = 0
                file_duration_min = 0
            else:
                # Calculate file-based duration
                earliest = min(timestamps)
                latest = max(timestamps)
                file_duration_ms = latest - earliest
                file_duration_min = file_duration_ms / (1000 * 60)

            print(f"File-based duration: {file_duration_min:.2f} minutes ({file_duration_ms/1000:.1f} seconds)")
            print(f"Total data size: {total_size/(1024*1024):.2f} MB")

            # Choose the appropriate duration for the session
            selected_duration_min = None
            clean_data_rate = None
            clean_classification = None
            file_data_rate = None
            file_classification = None

            # Calculate clean data rate if available
            if clean_duration_ms and clean_duration_ms > 0:
                clean_data_rate = calculate_data_rate(total_size, clean_duration_ms)
                clean_classification = classify_data_rate(clean_data_rate)
                print(f"Clean data rate: {clean_data_rate:.2f} KB/s - Classification: {clean_classification}")

                # Use clean duration if it's over 10 minutes
                if clean_duration_min > 10:
                    selected_duration_min = clean_duration_min

            # Always calculate file-based data rate
            file_data_rate = calculate_data_rate(total_size, file_duration_ms)
            file_classification = classify_data_rate(file_data_rate)
            print(f"File-based data rate: {file_data_rate:.2f} KB/s - Classification: {file_classification}")

            # If clean duration wasn't valid, use file-based duration if it's over 10 minutes
            if selected_duration_min is None and file_duration_min > 10:
                selected_duration_min = file_duration_min

            # If we have a valid duration, add it to our totals
            if selected_duration_min is not None:
                print(f"Session {session_id}: {selected_duration_min:.2f} minutes - adding to total")
                session_durations.append(selected_duration_min)
                valid_sessions += 1

                # Store session details for the API
                session_details.append({
                    "id": session_id,
                    "file_duration_min": file_duration_min,
                    "clean_duration_min": clean_duration_min,  # Always use clean_duration_min (which is now 0 instead of None)
                    "total_size_mb": total_size/(1024*1024),
                    "clean_data_rate": clean_data_rate if clean_data_rate is not None else 0,
                    "file_data_rate": file_data_rate,
                    "classification": clean_classification or file_classification,
                    "timestamp": datetime.now().isoformat()
                })
            else:
                print(f"Session {session_id} duration is less than 10 minutes, skipping")

        # Calculate total hours
        total_minutes = sum(session_durations)
        hours = total_minutes / 60

        print(f"Found {valid_sessions} valid sessions (>10 min) out of {len(prefixes)} total")
        print(f"Total minutes: {total_minutes:.2f}, total hours: {hours:.2f}")

        # Calculate device-specific metrics using clean durations
        total_both_minutes = 0
        total_fnirs_only_minutes = 0
        total_eeg_minutes = 0

        for session in session_details:
            # Always use the clean duration (which now defaults to 0 rather than None)
            clean_minutes = session["clean_duration_min"]

            if session["classification"] == "both (fnirs+eeg)":
                total_both_minutes += clean_minutes
                # Both counts as fnirs and eeg
                total_fnirs_only_minutes += clean_minutes
            elif session["classification"] == "fnirs only":
                total_fnirs_only_minutes += clean_minutes

        # Total EEG time is the same as total both time for now (as per requirements)
        total_eeg_minutes = total_both_minutes

        # Print detailed metrics
        print("\n--- DEVICE-SPECIFIC CLEAN TIME METRICS ---")
        print(f"Total clean time with both fnirs+eeg: {total_both_minutes:.2f} minutes ({total_both_minutes/60:.2f} hours)")
        print(f"Total clean time with fnirs: {total_fnirs_only_minutes:.2f} minutes ({total_fnirs_only_minutes/60:.2f} hours)")
        print(f"Total clean time with eeg: {total_eeg_minutes:.2f} minutes ({total_eeg_minutes/60:.2f} hours)")

        # Report on changes since last run
        if len(session_details) != previous_session_count:
            print(f"\nSession count changed from {previous_session_count} to {len(session_details)}")

            # Print a summary of device classifications
            classifications = {}
            for session in session_details:
                classification = session["classification"]
                classifications[classification] = classifications.get(classification, 0) + 1

            print("\nDevice classification summary:")
            for classification, count in classifications.items():
                print(f"  - {classification}: {count} sessions")

        total_hours = hours
        last_updated = datetime.now()

        return hours

    except Exception as e:
        print(f"Error calculating total hours: {e}")
        return 0
