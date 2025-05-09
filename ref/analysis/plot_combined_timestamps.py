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
    return datetime.fromtimestamp(ms/1000).strftime('%Y-%m-%d %H:%M:%S')

def evenly_spaced_indices(total_length, num_samples):
    """Get evenly spaced indices from a range of values"""
    if num_samples >= total_length:
        return np.arange(total_length)
    
    # Calculate the step size for evenly spaced samples
    step = total_length / num_samples
    
    # Generate indices
    indices = np.floor(np.arange(0, total_length, step)).astype(int)
    
    # Ensure we don't exceed the array length
    indices = indices[:num_samples]
    
    return indices

def plot_combined_timestamps(h5_file_path, max_samples=2000, output_prefix=''):
    """Analyze and plot timestamp data from an H5 file - only produces combined plot"""
    with h5py.File(h5_file_path, 'r') as f:
        # Check which modalities are available
        eeg_available = 'devices/eeg/timestamps' in f
        fnirs_available = 'devices/fnirs/timestamps' in f
        audio_available = 'audio/timestamps' in f
        
        # Check if we have at least one modality
        if not any([eeg_available, fnirs_available, audio_available]):
            print("Error: No timestamp datasets found in H5 file")
            return
            
        # Report which modalities are available
        print(f"Available modalities: EEG: {eeg_available}, fNIRS: {fnirs_available}, Audio: {audio_available}")
        
        # Read timestamp data for available modalities
        if eeg_available:
            eeg_timestamps = f['devices/eeg/timestamps'][()]
            print(f"EEG timestamps shape: {eeg_timestamps.shape}")
        else:
            eeg_timestamps = None
        
        if fnirs_available:
            fnirs_timestamps = f['devices/fnirs/timestamps'][()]
            print(f"fNIRS timestamps shape: {fnirs_timestamps.shape}")
        else:
            fnirs_timestamps = None
        
        if audio_available:
            audio_timestamps = f['audio/timestamps'][()]
            print(f"Audio timestamps shape: {audio_timestamps.shape}")
        else:
            audio_timestamps = None
        
        # Sample data if needed for available modalities
        if eeg_available:
            eeg_indices = evenly_spaced_indices(len(eeg_timestamps), min(max_samples, len(eeg_timestamps)))
            eeg_sampled = eeg_timestamps[eeg_indices]
            eeg_start_time = eeg_sampled[0, 1] / 1000  # Convert to seconds
            eeg_relative_times = np.array([(ts[1]/1000) for ts in eeg_sampled])
            eeg_offsets = eeg_sampled[:, 2]  # Third column contains offset
            eeg_device_server_diff = np.array([ts[0] - ts[1] for ts in eeg_sampled])
        
        if fnirs_available:
            fnirs_indices = evenly_spaced_indices(len(fnirs_timestamps), min(max_samples, len(fnirs_timestamps)))
            fnirs_sampled = fnirs_timestamps[fnirs_indices]
            fnirs_start_time = fnirs_sampled[0, 1] / 1000  # Convert to seconds
            fnirs_relative_times = np.array([(ts[1]/1000) for ts in fnirs_sampled])
            fnirs_offsets = fnirs_sampled[:, 2]  # Third column contains offset
            fnirs_device_server_diff = np.array([ts[0] - ts[1] for ts in fnirs_sampled])
        
        if audio_available:
            audio_indices = evenly_spaced_indices(len(audio_timestamps), min(max_samples, len(audio_timestamps)))
            audio_sampled = audio_timestamps[audio_indices]
            audio_start_time = audio_sampled[0, 1] / 1000  # Convert to seconds
            audio_relative_times = np.array([(ts[1]/1000) for ts in audio_sampled])
            # Audio has client and server timestamps in columns 0 and 1
            audio_client_server_diff = np.array([ts[0] - ts[1] for ts in audio_sampled])
        
        # Calculate common start time for relative timing
        start_times = []
        if eeg_available:
            start_times.append(eeg_start_time)
        if fnirs_available:
            start_times.append(fnirs_start_time)
        if audio_available:
            start_times.append(audio_start_time)
            
        # Use the earliest time as the reference point
        start_time = min(start_times)
        
        # Calculate relative times from common start time
        if eeg_available:
            eeg_relative_times = eeg_relative_times - start_time
        if fnirs_available:
            fnirs_relative_times = fnirs_relative_times - start_time
        if audio_available:
            audio_relative_times = audio_relative_times - start_time
        
        # Create output filename with prefix if provided
        combined_plot_filename = f'{output_prefix}combined_offsets.png' if output_prefix else 'combined_offsets.png'
        
        # Count available modalities for plotting
        available_modalities = sum([eeg_available, fnirs_available, audio_available])
        
        if available_modalities == 0:
            print("No modalities available for plotting")
            return
        
        # Create combined offset/diff plot
        plt.figure(figsize=(14, 6))
        
        if eeg_available:
            plt.plot(eeg_relative_times, eeg_offsets, 'b.', alpha=0.5, label='EEG Offsets')
        
        if fnirs_available:
            plt.plot(fnirs_relative_times, fnirs_offsets, 'r.', alpha=0.5, label='fNIRS Offsets')
        
        if audio_available:
            plt.plot(audio_relative_times, audio_client_server_diff, 'g.', alpha=0.5, label='Audio Client-Server Diff')
            
        plt.xlabel('Time (seconds from start)')
        plt.ylabel('Offset/Difference (ms)')
        plt.title('Combined Offsets and Time Differences')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(combined_plot_filename)
        print(f"Saved plot: {combined_plot_filename}")
        
        # Return statistics for all available modalities
        stats = {}
        
        if eeg_available:
            stats['eeg_stats'] = {
                'min_offset': np.min(eeg_offsets),
                'max_offset': np.max(eeg_offsets),
                'mean_offset': np.mean(eeg_offsets),
                'std_offset': np.std(eeg_offsets)
            }
            print("\nEEG Timestamp Statistics:")
            print(f"  Min offset: {np.min(eeg_offsets)}")
            print(f"  Max offset: {np.max(eeg_offsets)}")
            print(f"  Mean offset: {np.mean(eeg_offsets)}")
            print(f"  Std dev offset: {np.std(eeg_offsets)}")
        
        if fnirs_available:
            stats['fnirs_stats'] = {
                'min_offset': np.min(fnirs_offsets),
                'max_offset': np.max(fnirs_offsets),
                'mean_offset': np.mean(fnirs_offsets),
                'std_offset': np.std(fnirs_offsets)
            }
            print("\nfNIRS Timestamp Statistics:")
            print(f"  Min offset: {np.min(fnirs_offsets)}")
            print(f"  Max offset: {np.max(fnirs_offsets)}")
            print(f"  Mean offset: {np.mean(fnirs_offsets)}")
            print(f"  Std dev offset: {np.std(fnirs_offsets)}")
        
        if audio_available:
            stats['audio_stats'] = {
                'min_diff': np.min(audio_client_server_diff),
                'max_diff': np.max(audio_client_server_diff),
                'mean_diff': np.mean(audio_client_server_diff),
                'std_diff': np.std(audio_client_server_diff)
            }
            print("\nAudio Timestamp Statistics:")
            print(f"  Min client-server diff: {np.min(audio_client_server_diff)}")
            print(f"  Max client-server diff: {np.max(audio_client_server_diff)}")
            print(f"  Mean client-server diff: {np.mean(audio_client_server_diff)}")
            print(f"  Std dev client-server diff: {np.std(audio_client_server_diff)}")
        
        return stats

def main():
    parser = argparse.ArgumentParser(description='Generate combined timestamp plots from H5 files')
    parser.add_argument('--session', type=str, required=True,
                      help='Session name to analyze')
    parser.add_argument('--samples', type=int, default=2000,
                      help='Number of samples to use in plots (default: 2000)')
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
            # Generate combined timestamps plot
            plot_combined_timestamps(temp_path, args.samples, args.output_prefix)
        else:
            print("Failed to download H5 file")
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    main()