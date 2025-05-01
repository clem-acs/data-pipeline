#!/usr/bin/env python3
"""
Test script for EEG preprocessing functions.

This script:
1. Downloads a session H5 file from S3
2. Applies EEG preprocessing
3. Expands EEG timestamps to match processed data
4. Verifies the processing completes without errors
5. Displays statistics about before and after data

Usage:
    python test_eeg_preprocessing.py <session_id>
"""

import os
import sys
import argparse
import tempfile
import time
import boto3
import h5py
import numpy as np
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transforms.neural_processing.eeg_preprocessing import preprocess_eeg, expand_eeg_timestamps
from utils.aws import get_aws_credentials

# Constants
BUCKET_NAME = "conduit-data-dev"
PREFIX = "data-collector/new-sessions/"


def init_s3_client():
    """Initialize S3 client with AWS credentials from environment"""
    # Load environment variables
    load_dotenv()
    
    # Get AWS credentials
    credentials = get_aws_credentials()
    aws_access_key_id = credentials["aws_access_key_id"]
    aws_secret_access_key = credentials["aws_secret_access_key"]
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


def find_session_file(s3_client, session_id):
    """Find the main H5 file for a specific session ID"""
    session_prefix = f"{PREFIX}{session_id}/"
    print(f"Looking for H5 files in s3://{BUCKET_NAME}/{session_prefix}")
    
    paginator = s3_client.get_paginator('list_objects_v2')
    main_h5_file = None
    
    for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix=session_prefix):
        if 'Contents' in page:
            for obj in page['Contents']:
                key = obj['Key']
                file_name = key.split('/')[-1]
                
                # Look for main H5 file in session root
                if file_name.endswith('.h5'):
                    if '/files/' not in key:  # Skip files in subdirectories
                        main_h5_file = key
                        break
                        
    return main_h5_file


def download_h5_file(s3_client, s3_path, local_path):
    """Download an H5 file from S3 to a local path"""
    try:
        print(f"Downloading {s3_path} to {local_path}")
        start_time = time.time()
        s3_client.download_file(BUCKET_NAME, s3_path, local_path)
        download_time = time.time() - start_time
        
        file_size_mb = os.path.getsize(local_path) / (1024 * 1024)
        print(f"Downloaded {file_size_mb:.2f} MB in {download_time:.2f} seconds")
        return True
    except Exception as e:
        print(f"Error downloading {s3_path}: {str(e)}")
        return False


def extract_eeg_data(h5_path):
    """Extract EEG data and timestamps from an H5 file"""
    try:
        with h5py.File(h5_path, 'r') as f:
            # Print summary of key datasets
            print("\nKey data in H5 file:")
            if 'devices/eeg/frames_data' in f:
                print(f"  EEG: {f['devices/eeg/frames_data'].shape}")
            if 'devices/fnirs/frames_data' in f:
                print(f"  fNIRS: {f['devices/fnirs/frames_data'].shape}")
            if 'events/display/data' in f:
                print(f"  Display events: {len(f['events/display/data'])}")
            
            # Directly access EEG data at the known location
            eeg_data = None
            eeg_timestamps = None
            
            if 'devices/eeg/frames_data' in f:
                eeg_data = f['devices/eeg/frames_data'][:]
                print(f"\nFound EEG data with shape {eeg_data.shape}")
                
                # Try to get timestamps
                if 'devices/eeg/timestamps' in f:
                    eeg_timestamps = f['devices/eeg/timestamps'][:]
                    print(f"Found EEG timestamps with shape {eeg_timestamps.shape}")
                
                metadata = {
                    'sample_rate': 250.0,  # Default sample rate
                    'data_format': '3D frames',
                    'frame_size': eeg_data.shape[2],
                    'channels': eeg_data.shape[1]
                }
                
                # Try to extract sample rate if available
                if 'devices/eeg/sample_rate' in f:
                    metadata['sample_rate'] = float(f['devices/eeg/sample_rate'][()])
                
                return eeg_data, eeg_timestamps, metadata
            else:
                print("EEG data not found at expected location: devices/eeg/frames_data")
                return None, None, None
            
    except Exception as e:
        print(f"Error extracting EEG data: {e}")
        return None, None, None


def test_preprocessing(eeg_data, eeg_timestamps, metadata):
    """Test both EEG preprocessing and timestamp expansion functions"""
    if eeg_data is None:
        print("No EEG data found to test preprocessing")
        return False
    
    try:
        print("\nTesting EEG preprocessing...")
        print(f"Input EEG data shape: {eeg_data.shape}")
        if eeg_timestamps is not None:
            print(f"Input EEG timestamps shape: {eeg_timestamps.shape}")
        
        # The data should already be in the correct (frames, channels, samples_per_frame) format
        input_data = eeg_data
        
        # Show initial metadata
        print(f"\nInput metadata:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")
        
        # Time the preprocessing
        start_time = time.time()
        processed_eeg, preprocessing_metadata = preprocess_eeg(input_data, metadata)
        processing_time = time.time() - start_time
        
        print(f"\nPreprocessing completed in {processing_time:.2f} seconds")
        print(f"Output EEG data shape: {processed_eeg.shape}")
        print(f"Output metadata:")
        for key, value in preprocessing_metadata.items():
            print(f"  {key}: {value}")
        
        # Calculate how many samples were in the original vs processed data
        original_total_samples = input_data.shape[0] * input_data.shape[2]
        processed_total_samples = processed_eeg.shape[0]
        
        print(f"\nSummary:")
        print(f"  Original data: {input_data.shape[0]} frames × {input_data.shape[2]} samples/frame = {original_total_samples} total samples")
        print(f"  Original channels: {input_data.shape[1]}")
        print(f"  Processed data: {processed_eeg.shape[0]} samples × {processed_eeg.shape[1]} channels")
        print(f"  Channels used: {preprocessing_metadata['channels_used']} (from original {input_data.shape[1]})")
        
        # Basic data quality checks
        print("\nData quality checks:")
        print(f"  Input data - min: {np.min(input_data)}, max: {np.max(input_data)}, mean: {np.mean(input_data):.6f}, std: {np.std(input_data):.6f}")
        print(f"  Output data - min: {np.min(processed_eeg)}, max: {np.max(processed_eeg)}, mean: {np.mean(processed_eeg):.6f}, std: {np.std(processed_eeg):.6f}")
        
        # Check for NaN values
        input_nans = np.isnan(input_data).sum()
        output_nans = np.isnan(processed_eeg).sum()
        print(f"  NaN values - input: {input_nans}, output: {output_nans}")
        
        # Check for inf values
        input_infs = np.isinf(input_data).sum()
        output_infs = np.isinf(processed_eeg).sum()
        print(f"  Inf values - input: {input_infs}, output: {output_infs}")
        
        # Check for reasonable range of values for EEG (typically -100 to 100 µV)
        if np.max(processed_eeg) > 500 or np.min(processed_eeg) < -500:
            print("  WARNING: Processed data contains unusually large values outside typical EEG range")
        
        # Test timestamp expansion if timestamps are available
        if eeg_timestamps is not None:
            print("\n\nTesting EEG timestamp expansion...")
            print(f"Input timestamps shape: {eeg_timestamps.shape}")
            
            # Time the timestamp expansion
            start_time = time.time()
            expanded_timestamps, timestamp_metadata = expand_eeg_timestamps(eeg_timestamps, metadata)
            expansion_time = time.time() - start_time
            
            print(f"Timestamp expansion completed in {expansion_time:.2f} seconds")
            print(f"Expanded timestamps shape: {expanded_timestamps.shape}")
            print(f"Timestamp metadata:")
            for key, value in timestamp_metadata.items():
                print(f"  {key}: {value}")
            
            # Verify timestamps match processed data length
            if expanded_timestamps.shape[0] == processed_eeg.shape[0]:
                print(f"\nSUCCESS: Expanded timestamps match processed EEG data length ({expanded_timestamps.shape[0]})")
            else:
                print(f"\nWARNING: Expanded timestamps length ({expanded_timestamps.shape[0]}) does not match processed EEG data length ({processed_eeg.shape[0]})")
            
            # Show some sample timestamps
            print("\nSample of original timestamps (first 3):")
            for i in range(min(3, eeg_timestamps.shape[0])):
                print(f"  {i}: {eeg_timestamps[i]}")
                
            print("\nSample of expanded timestamps (first 3 from different frames):")
            for i in range(0, min(3*metadata.get('frame_size', 15), expanded_timestamps.shape[0]), metadata.get('frame_size', 15)):
                print(f"  {i}: {expanded_timestamps[i]}")
        
        return True
    except Exception as e:
        print(f"Error during preprocessing test: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Test EEG preprocessing on a specific session")
    parser.add_argument("session_id", help="The session ID to process")
    args = parser.parse_args()
    
    session_id = args.session_id
    
    # Initialize S3 client
    print("Initializing S3 client...")
    s3_client = init_s3_client()
    
    # Find the session file
    h5_key = find_session_file(s3_client, session_id)
    if not h5_key:
        print(f"No H5 file found for session ID {session_id}")
        return 1
    
    print(f"Found H5 file: {h5_key}")
    
    # Create a temporary file for downloading
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as temp_file:
        temp_path = temp_file.name
    
    try:
        # Download the H5 file
        if not download_h5_file(s3_client, h5_key, temp_path):
            print("Failed to download H5 file")
            return 1
        
        # Extract EEG data and timestamps
        eeg_data, eeg_timestamps, metadata = extract_eeg_data(temp_path)
        if eeg_data is None:
            print("Failed to extract EEG data")
            return 1
        
        # Test preprocessing and timestamp expansion
        success = test_preprocessing(eeg_data, eeg_timestamps, metadata)
        
        if success:
            print("\nEEG preprocessing and timestamp expansion tests completed successfully!")
            return 0
        else:
            print("\nEEG processing tests failed.")
            return 1
    
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)
            print(f"Cleaned up temporary file: {temp_path}")


if __name__ == "__main__":
    sys.exit(main())