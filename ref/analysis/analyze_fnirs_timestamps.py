#!/usr/bin/env python
import os
import tempfile
import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt
import boto3
from datetime import datetime

# S3 bucket and prefixes
BUCKET_NAME = 'conduit-data-dev'
NEW_SESSIONS_PREFIX = 'data-collector/new-sessions/'
TEST_RUNS_PREFIX = 'data-collector/raw-data/test-runs/'

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

def download_h5_file(s3_client, s3_path, local_path):
    """Download an H5 file from S3 to a local path"""
    try:
        print(f"Downloading {s3_path} to {local_path}...")
        s3_client.download_file(BUCKET_NAME, s3_path, local_path)
        return True
    except Exception as e:
        print(f"Error downloading {s3_path}: {e}")
        return False

def ms_to_datetime_str(ms):
    """Convert milliseconds to datetime string"""
    return datetime.fromtimestamp(ms/1000).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

def analyze_fnirs_timestamps(h5_file_path, output_prefix=''):
    """Analyze and plot fNIRS timestamp differences between adjacent frames"""
    with h5py.File(h5_file_path, 'r') as f:
        # Check if the required datasets exist
        if 'devices/fnirs/timestamps' not in f:
            print("Error: fNIRS timestamp dataset not found in H5 file")
            return
        
        # Read fNIRS timestamps data
        fnirs_timestamps = f['devices/fnirs/timestamps'][()]
        
        print(f"fNIRS timestamps shape: {fnirs_timestamps.shape}")
        
        # Extract the timestamps (column 1 contains server timestamps)
        fnirs_server_timestamps = fnirs_timestamps[:, 1]
        
        # Calculate the differences between adjacent timestamps (in milliseconds)
        # This gives us the time between consecutive frames
        timestamp_diffs = np.diff(fnirs_server_timestamps)
        
        # Convert absolute timestamps to relative times in seconds from start
        start_time = fnirs_server_timestamps[0]
        relative_times = (fnirs_server_timestamps - start_time) / 1000.0  # Convert to seconds
        
        # The times for the differences are the times of the second frame in each pair
        diff_times = relative_times[1:]
        
        # Create output filenames with prefix if provided
        diffs_plot_filename = f'{output_prefix}fnirs_timestamp_diffs.png' if output_prefix else 'fnirs_timestamp_diffs.png'
        hist_plot_filename = f'{output_prefix}fnirs_timestamp_diffs_histogram.png' if output_prefix else 'fnirs_timestamp_diffs_histogram.png'
        
        # 1. Plot timestamp differences over time
        plt.figure(figsize=(14, 8))
        plt.plot(diff_times, timestamp_diffs, 'r.', alpha=0.7)
        plt.xlabel('Time (seconds from start)')
        plt.ylabel('Time Difference (ms)')
        plt.title('fNIRS Timestamp Differences Between Adjacent Frames')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(diffs_plot_filename)
        print(f"Saved plot: {diffs_plot_filename}")
        
        # 2. Histogram of timestamp differences
        plt.figure(figsize=(14, 8))
        plt.hist(timestamp_diffs, bins=50, alpha=0.7, color='red')
        plt.xlabel('Timestamp Difference (ms)')
        plt.ylabel('Count')
        plt.title('Distribution of fNIRS Timestamp Differences')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(hist_plot_filename)
        print(f"Saved histogram: {hist_plot_filename}")
        
        # 3. Zoomed in sections to see patterns
        # First 100 samples
        zoom_plot_filename = f'{output_prefix}fnirs_timestamp_diffs_zoom.png' if output_prefix else 'fnirs_timestamp_diffs_zoom.png'
        plt.figure(figsize=(14, 8))
        
        sample_limit = min(100, len(diff_times))
        plt.plot(diff_times[:sample_limit], timestamp_diffs[:sample_limit], 'ro-', markersize=4)
        plt.xlabel('Time (seconds from start)')
        plt.ylabel('Time Difference (ms)')
        plt.title('fNIRS Timestamp Differences (First 100 Samples)')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(zoom_plot_filename)
        print(f"Saved zoomed plot: {zoom_plot_filename}")
        
        # Calculate statistics
        print("\nfNIRS Timestamp Difference Statistics:")
        print(f"  Total frames: {len(fnirs_timestamps)}")
        print(f"  First timestamp: {ms_to_datetime_str(fnirs_server_timestamps[0])}")
        print(f"  Last timestamp: {ms_to_datetime_str(fnirs_server_timestamps[-1])}")
        print(f"  Session duration: {(fnirs_server_timestamps[-1] - fnirs_server_timestamps[0])/1000:.2f} seconds")
        print(f"  Min difference: {np.min(timestamp_diffs):.2f} ms")
        print(f"  Max difference: {np.max(timestamp_diffs):.2f} ms")
        print(f"  Mean difference: {np.mean(timestamp_diffs):.2f} ms")
        print(f"  Median difference: {np.median(timestamp_diffs):.2f} ms")
        print(f"  Std dev difference: {np.std(timestamp_diffs):.2f} ms")
        
        # Calculate the expected average sampling rate
        avg_sampling_rate = 1000 / np.mean(timestamp_diffs)  # Hz
        print(f"  Estimated average sampling rate: {avg_sampling_rate:.2f} Hz")
        
        # Check for outliers (differences that are significantly different from the median)
        median_diff = np.median(timestamp_diffs)
        outlier_threshold = 3 * np.std(timestamp_diffs)
        outliers = np.where(np.abs(timestamp_diffs - median_diff) > outlier_threshold)[0]
        
        if len(outliers) > 0:
            print(f"\nFound {len(outliers)} outliers in timestamp differences (> 3 std devs from median)")
            print("Top 10 outliers:")
            
            # Sort outliers by the absolute difference from median
            sorted_outliers = sorted(outliers, key=lambda i: abs(timestamp_diffs[i] - median_diff), reverse=True)
            
            for i in sorted_outliers[:10]:
                frame_idx = i + 1  # +1 because diff[i] is between timestamp[i] and timestamp[i+1]
                diff_value = timestamp_diffs[i]
                timestamp = fnirs_server_timestamps[frame_idx]
                print(f"  Frame {frame_idx}: {diff_value:.2f} ms at {ms_to_datetime_str(timestamp)}")
        
        # Look for common patterns - get the unique differences and their counts
        unique_diffs, diff_counts = np.unique(np.round(timestamp_diffs, 2), return_counts=True)
        
        # Sort by count (most common first)
        sort_indices = np.argsort(-diff_counts)
        sorted_unique_diffs = unique_diffs[sort_indices]
        sorted_diff_counts = diff_counts[sort_indices]
        
        print("\nMost common timestamp differences:")
        for i in range(min(10, len(sorted_unique_diffs))):
            diff_value = sorted_unique_diffs[i]
            count = sorted_diff_counts[i]
            percentage = 100 * count / len(timestamp_diffs)
            print(f"  {diff_value:.2f} ms: {count} occurrences ({percentage:.2f}%)")
        
        # Return stats for reference
        return {
            'total_frames': len(fnirs_timestamps),
            'session_duration_ms': fnirs_server_timestamps[-1] - fnirs_server_timestamps[0],
            'min_diff': np.min(timestamp_diffs),
            'max_diff': np.max(timestamp_diffs),
            'mean_diff': np.mean(timestamp_diffs),
            'median_diff': np.median(timestamp_diffs),
            'std_diff': np.std(timestamp_diffs),
            'sampling_rate_hz': avg_sampling_rate,
            'outlier_count': len(outliers),
            'common_diffs': list(zip(sorted_unique_diffs[:10], sorted_diff_counts[:10]))
        }

def main():
    parser = argparse.ArgumentParser(description='Analyze fNIRS timestamp differences')
    parser.add_argument('--session', type=str, required=True,
                      help='Session name to analyze')
    parser.add_argument('--output-prefix', type=str, default='',
                      help='Prefix for output files (default: none)')
    parser.add_argument('--exact-name', action='store_true',
                      help='Use exact session name (no substring matching)')
    parser.add_argument('--test', action='store_true',
                      help='Use test-runs directory instead of new-sessions')
    
    args = parser.parse_args()
    
    # Initialize S3 client
    s3_client = init_s3_client()
    
    # Select the appropriate prefix based on the --test flag
    PREFIX = TEST_RUNS_PREFIX if args.test else NEW_SESSIONS_PREFIX
    print(f"Using S3 path: s3://{BUCKET_NAME}/{PREFIX}")
    
    # Find the session folder
    if args.exact_name:
        session_path = f"{PREFIX}{args.session}"
        found_session = True
    else:
        # Search for the session by name
        paginator = s3_client.get_paginator('list_objects_v2')
        found_session = False
        session_path = None
        
        for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix=PREFIX, Delimiter='/'):
            if 'CommonPrefixes' in page:
                for prefix_obj in page['CommonPrefixes']:
                    folder_path = prefix_obj['Prefix']
                    folder_name = folder_path.rstrip('/').split('/')[-1]
                    
                    if args.session in folder_name:
                        session_path = folder_path
                        print(f"Found matching session: {folder_name}")
                        found_session = True
                        break
                        
            if found_session:
                break
    
    if not found_session or not session_path:
        print(f"No session found matching: {args.session}")
        return
    
    # Get session name from path
    session_name = session_path.rstrip('/').split('/')[-1]
    print(f"Analyzing session: {session_name}")
    
    # Find the main H5 file
    main_h5_file = find_main_h5_file(s3_client, session_path, session_name)
    
    if not main_h5_file:
        print(f"No main H5 file found for session: {session_name}")
        return
    
    # Create a temporary file to download to
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as temp_file:
        temp_path = temp_file.name
    
    try:
        # Download the file
        if download_h5_file(s3_client, main_h5_file, temp_path):
            # Analyze fNIRS timestamps
            analyze_fnirs_timestamps(temp_path, args.output_prefix)
        else:
            print("Failed to download H5 file")
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    main()