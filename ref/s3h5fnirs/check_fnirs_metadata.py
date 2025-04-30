#!/usr/bin/env python3
"""
Script to extract and display metadata for the fNIRS group in H5 files.

This script:
1. Downloads an H5 file from S3
2. Extracts all metadata (attributes) associated with the devices/fnirs group
3. Displays the metadata in a readable format

Usage:
python check_fnirs_metadata.py --key "curated-h5/filename.h5"
"""

import os
import sys
import argparse
import logging
import tempfile
import h5py
import numpy as np
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("fnirs_metadata")


def extract_h5_metadata(file_path):
    """Extract metadata from an H5 file.
    
    Args:
        file_path: Path to the H5 file
        
    Returns:
        Dictionary with metadata information
    """
    logger.info(f"Extracting metadata from {file_path}")
    
    metadata = {
        'file_path': file_path,
        'file_name': os.path.basename(file_path),
        'file_size_mb': os.path.getsize(file_path) / (1024 * 1024),
        'root_attributes': {},
        'fnirs_attributes': {},
        'fnirs_children': [],
        'fnirs_datasets': [],
        'other_devices': []
    }
    
    try:
        with h5py.File(file_path, 'r') as f:
            # Extract root attributes
            for key in f.attrs:
                metadata['root_attributes'][key] = f.attrs[key]
            
            # Check for devices group
            if 'devices' in f:
                # Check for other device types
                for device in f['devices']:
                    if device != 'fnirs':
                        metadata['other_devices'].append(device)
                
                # Check for fNIRS group
                if 'fnirs' in f['devices']:
                    fnirs_group = f['devices/fnirs']
                    
                    # Extract fNIRS attributes
                    for key in fnirs_group.attrs:
                        metadata['fnirs_attributes'][key] = fnirs_group.attrs[key]
                    
                    # List children of fNIRS group
                    for name in fnirs_group:
                        child_path = f"devices/fnirs/{name}"
                        child = fnirs_group[name]
                        
                        if isinstance(child, h5py.Group):
                            # It's a subgroup
                            subgroup_info = {
                                'name': name,
                                'path': child_path,
                                'type': 'group',
                                'attributes': {key: child.attrs[key] for key in child.attrs}
                            }
                            metadata['fnirs_children'].append(subgroup_info)
                        elif isinstance(child, h5py.Dataset):
                            # It's a dataset
                            dataset_info = {
                                'name': name,
                                'path': child_path,
                                'type': 'dataset',
                                'shape': child.shape,
                                'dtype': str(child.dtype),
                                'size_mb': child.size * child.dtype.itemsize / (1024 * 1024),
                                'attributes': {key: child.attrs[key] for key in child.attrs}
                            }
                            metadata['fnirs_datasets'].append(dataset_info)
                            
                            # For timestamps dataset, get some extra info
                            if name == 'timestamps' and child.size > 0:
                                timestamps = child[:]
                                if len(timestamps) > 1:
                                    # Calculate sample rate
                                    time_diffs = np.diff(timestamps[:100, 0] if timestamps.ndim > 1 else timestamps[:100])
                                    mean_diff = np.mean(time_diffs)
                                    
                                    dataset_info['first_timestamp'] = float(timestamps[0][0] if timestamps.ndim > 1 else timestamps[0])
                                    dataset_info['last_timestamp'] = float(timestamps[-1][0] if timestamps.ndim > 1 else timestamps[-1])
                                    dataset_info['duration_sec'] = float(timestamps[-1][0] - timestamps[0][0] if timestamps.ndim > 1 else timestamps[-1] - timestamps[0])
                                    dataset_info['mean_sample_interval'] = float(mean_diff)
                                    dataset_info['estimated_sample_rate'] = float(1 / mean_diff if mean_diff > 0 else 0)
            else:
                logger.warning("No 'devices' group found in the H5 file")
    
    except Exception as e:
        logger.error(f"Error extracting metadata: {e}", exc_info=True)
    
    return metadata


def download_and_extract_metadata(bucket, key):
    """Download an H5 file from S3 and extract its metadata.
    
    Args:
        bucket: S3 bucket name
        key: S3 object key
        
    Returns:
        Dictionary with metadata information
    """
    # Add parent directory to path to import utils
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
    from utils.aws import init_s3_client
    
    logger.info(f"Downloading and extracting metadata from s3://{bucket}/{key}")
    
    # Create temporary directory for download
    temp_dir = tempfile.mkdtemp(prefix="fnirs_metadata_")
    local_path = os.path.join(temp_dir, os.path.basename(key))
    
    try:
        # Initialize S3 client
        s3_client = init_s3_client()
        
        # Download the file
        logger.info(f"Downloading to {local_path}")
        s3_client.download_file(bucket, key, local_path)
        logger.info(f"Downloaded {os.path.getsize(local_path)/1024/1024:.2f} MB")
        
        # Extract metadata
        metadata = extract_h5_metadata(local_path)
        
        # Clean up temporary file
        os.remove(local_path)
        os.rmdir(temp_dir)
        
        return metadata
    
    except Exception as e:
        logger.error(f"Error downloading or extracting metadata: {e}", exc_info=True)
        return {
            'error': str(e),
            'file_path': f"s3://{bucket}/{key}"
        }


def print_metadata_summary(metadata):
    """Print a summary of the metadata.
    
    Args:
        metadata: Dictionary with metadata information
    """
    print("\n" + "="*80)
    print(f"FNIRS METADATA SUMMARY")
    print("="*80)
    
    if 'error' in metadata:
        print(f"Error: {metadata['error']}")
        return
    
    print(f"File: {metadata['file_name']}")
    print(f"Size: {metadata['file_size_mb']:.2f} MB")
    
    # Print root attributes if any
    if metadata['root_attributes']:
        print("\nRoot Attributes:")
        for key, value in metadata['root_attributes'].items():
            print(f"  {key}: {value}")
    
    # Print fNIRS attributes
    if metadata['fnirs_attributes']:
        print("\nfNIRS Group Attributes:")
        for key, value in metadata['fnirs_attributes'].items():
            print(f"  {key}: {value}")
    
    # Print other devices
    if metadata['other_devices']:
        print("\nOther Device Types:")
        for device in metadata['other_devices']:
            print(f"  - {device}")
    
    # Print fNIRS datasets
    if metadata['fnirs_datasets']:
        print("\nfNIRS Datasets:")
        for dataset in metadata['fnirs_datasets']:
            print(f"  {dataset['name']}:")
            print(f"    Shape: {dataset['shape']}")
            print(f"    Size: {dataset['size_mb']:.2f} MB")
            print(f"    Type: {dataset['dtype']}")
            
            # Print timestamps-specific info if available
            if 'first_timestamp' in dataset:
                print(f"    First Timestamp: {dataset['first_timestamp']}")
                print(f"    Last Timestamp: {dataset['last_timestamp']}")
                print(f"    Duration: {dataset['duration_sec']:.2f} seconds")
                print(f"    Estimated Sample Rate: {dataset['estimated_sample_rate']:.2f} Hz")
            
            # Print attributes if any
            if dataset['attributes']:
                print("    Attributes:")
                for key, value in dataset['attributes'].items():
                    print(f"      {key}: {value}")
    
    # Print fNIRS child groups
    if metadata['fnirs_children']:
        print("\nfNIRS Child Groups:")
        for child in metadata['fnirs_children']:
            print(f"  {child['name']}:")
            
            # Print attributes if any
            if child['attributes']:
                print("    Attributes:")
                for key, value in child['attributes'].items():
                    print(f"      {key}: {value}")
    
    print("="*80)


def main():
    """Main function to run the script."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Extract and display fNIRS metadata from H5 files")
    parser.add_argument("--bucket", default="conduit-data-dev", help="S3 bucket name")
    parser.add_argument("--key", required=True, help="S3 object key")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()
    
    # Set log level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Download and extract metadata
    metadata = download_and_extract_metadata(args.bucket, args.key)
    
    # Print metadata summary
    print_metadata_summary(metadata)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())