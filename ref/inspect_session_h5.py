#!/usr/bin/env python
import os
import sys
import boto3
import tempfile
import h5py
import argparse
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# S3 bucket details
BUCKET_NAME = "conduit-data-dev"
DEFAULT_PREFIX = "data-collector/new-sessions/"

def init_s3_client():
    """Initialize S3 client with AWS credentials from environment"""
    # Check if credentials are available
    aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    region_name = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
    
    if not aws_access_key_id or not aws_secret_access_key:
        print("WARNING: AWS credentials not found in environment variables.")
        print("Make sure your .env file contains AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY")
    
    return boto3.client(
        's3',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=region_name
    )

def find_session_path(s3_client, session_id):
    """Find the full S3 path for a session by its ID"""
    paginator = s3_client.get_paginator('list_objects_v2')
    
    for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix=DEFAULT_PREFIX):
        if 'Contents' in page:
            for obj in page['Contents']:
                key = obj['Key']
                if session_id in key:
                    # Extract the session path (directory)
                    path_parts = key.split('/')
                    # Find the session_id in the path
                    for i, part in enumerate(path_parts):
                        if session_id in part:
                            # Return the path up to and including the session_id directory
                            session_path = '/'.join(path_parts[:i+1]) + '/'
                            return session_path
    
    return None

def find_main_h5_file(s3_client, session_path):
    """Find the main H5 file in a session directory"""
    paginator = s3_client.get_paginator('list_objects_v2')
    h5_files = []
    
    for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix=session_path):
        if 'Contents' in page:
            for obj in page['Contents']:
                file_key = obj['Key']
                file_name = file_key.split('/')[-1]
                
                # Only consider direct h5 files in the session directory, not in subdirectories
                if file_name.endswith('.h5') and file_key.count('/') == session_path.count('/'):
                    h5_files.append((file_key, obj['Size']))
    
    # Sort files by name and return the first one
    if h5_files:
        h5_files.sort(key=lambda x: x[0])
        return h5_files[0][0]
    
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

def examine_h5_file(file_path):
    """Examine the structure of an H5 file including EEG data"""
    try:
        with h5py.File(file_path, 'r') as f:
            print("\n====== H5 File Structure ======")
            
            # Print root attributes
            print("\nRoot level attributes:")
            for attr_name, attr_value in f.attrs.items():
                print(f"  {attr_name}: {attr_value}")
            
            # Print top-level groups and datasets
            print("\nTop-level groups and datasets:")
            for name in f:
                if isinstance(f[name], h5py.Group):
                    print(f"  Group: {name}")
                else:
                    print(f"  Dataset: {name} (shape: {f[name].shape}, dtype: {f[name].dtype})")
            
            # Look for EEG data
            eeg_paths = []
            def find_eeg_data(name, obj):
                if 'eeg' in name.lower():
                    if isinstance(obj, h5py.Dataset):
                        eeg_paths.append(name)
            
            # Traverse all groups to find EEG datasets
            f.visititems(find_eeg_data)
            
            if eeg_paths:
                print("\n====== EEG Data Found ======")
                for path in eeg_paths:
                    dataset = f[path]
                    print(f"\nEEG Dataset: {path}")
                    print(f"  Shape: {dataset.shape}")
                    print(f"  Data Type: {dataset.dtype}")
                    
                    # Print attributes if any
                    if len(dataset.attrs) > 0:
                        print("  Attributes:")
                        for attr_name, attr_value in dataset.attrs.items():
                            print(f"    {attr_name}: {attr_value}")
                    
                    # Print some sample data if small enough
                    if len(dataset.shape) > 0 and dataset.shape[0] > 0:
                        sample_size = min(5, dataset.shape[0])
                        print(f"  Sample data (first {sample_size} entries):")
                        sample = dataset[:sample_size]
                        for i, value in enumerate(sample):
                            if i < 5:  # Limit output
                                print(f"    {i}: {value}")
            else:
                print("\nNo EEG data found in the file")
                
            # Check for timestamps related to EEG
            timestamp_paths = []
            def find_timestamps(name, obj):
                if ('time' in name.lower() or 'timestamp' in name.lower()) and 'eeg' in name.lower():
                    if isinstance(obj, h5py.Dataset):
                        timestamp_paths.append(name)
            
            f.visititems(find_timestamps)
            
            if timestamp_paths:
                print("\n====== EEG Timestamps Found ======")
                for path in timestamp_paths:
                    dataset = f[path]
                    print(f"\nTimestamp Dataset: {path}")
                    print(f"  Shape: {dataset.shape}")
                    print(f"  Data Type: {dataset.dtype}")
                    
                    # Print some sample timestamps
                    if len(dataset.shape) > 0 and dataset.shape[0] > 0:
                        sample_size = min(5, dataset.shape[0])
                        print(f"  Sample timestamps (first {sample_size} entries):")
                        sample = dataset[:sample_size]
                        for i, value in enumerate(sample):
                            if i < 5:  # Limit output
                                print(f"    {i}: {value}")
            
            return eeg_paths
    except Exception as e:
        print(f"Error examining H5 file: {e}")
        return []

def parse_arguments():
    parser = argparse.ArgumentParser(description='Inspect H5 file for a specific session ID on S3')
    parser.add_argument('session_id', help='The session ID to find and inspect')
    parser.add_argument('--partial', action='store_true', help='Download only part of the file (for large files)')
    parser.add_argument('--max-size', type=int, default=100, help='Maximum size in MB to download if using partial download')
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    print(f"Looking for session with ID: {args.session_id}")
    s3_client = init_s3_client()
    
    # Find the session path
    session_path = find_session_path(s3_client, args.session_id)
    if not session_path:
        print(f"Error: Could not find session with ID {args.session_id}")
        sys.exit(1)
    
    print(f"Found session at: s3://{BUCKET_NAME}/{session_path}")
    
    # Find the main H5 file
    h5_file_path = find_main_h5_file(s3_client, session_path)
    if not h5_file_path:
        print(f"Error: Could not find any H5 files in the session directory")
        sys.exit(1)
    
    print(f"Found H5 file: {h5_file_path}")
    
    # Create a temporary file for downloading
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as temp_file:
        temp_path = temp_file.name
    
    try:
        # Download the file
        print(f"Downloading {'partial' if args.partial else 'complete'} H5 file...")
        if download_h5_file(s3_client, h5_file_path, temp_path, partial=args.partial, max_size_mb=args.max_size):
            # Examine the file structure focusing on EEG data
            eeg_paths = examine_h5_file(temp_path)
            
            if not eeg_paths:
                print("\nNo EEG data found in this file.")
            else:
                print(f"\nFound {len(eeg_paths)} EEG datasets in the file.")
        else:
            print("Failed to download the H5 file")
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    main()