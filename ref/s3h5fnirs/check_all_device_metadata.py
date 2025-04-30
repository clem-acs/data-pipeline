#!/usr/bin/env python3
"""
Script to extract and display all device metadata in H5 files.

This script:
1. Downloads an H5 file from S3
2. Extracts all metadata (attributes) associated with all device groups
3. Checks all groups in the file for metadata
4. Displays the metadata in a readable format

Usage:
python check_all_device_metadata.py --key "curated-h5/filename.h5"
"""

import os
import sys
import argparse
import logging
import tempfile
import h5py
import numpy as np
from pathlib import Path
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("device_metadata")


def extract_attributes(group):
    """Extract attributes from an HDF5 group.
    
    Args:
        group: h5py Group object
        
    Returns:
        Dictionary with attributes
    """
    attributes = {}
    for key in group.attrs:
        try:
            value = group.attrs[key]
            # Convert NumPy types to Python types for better display
            if isinstance(value, np.ndarray):
                value = value.tolist()
            elif isinstance(value, (np.int32, np.int64)):
                value = int(value)
            elif isinstance(value, (np.float32, np.float64)):
                value = float(value)
            elif isinstance(value, np.bool_):
                value = bool(value)
            elif isinstance(value, bytes):
                try:
                    value = value.decode('utf-8')
                except:
                    value = str(value)
            
            attributes[key] = value
        except Exception as e:
            attributes[key] = f"[Error reading attribute: {str(e)}]"
    
    return attributes


def extract_h5_structure(file_path):
    """Extract structure and metadata from an H5 file.
    
    Args:
        file_path: Path to the H5 file
        
    Returns:
        Dictionary with structure and metadata information
    """
    logger.info(f"Extracting structure and metadata from {file_path}")
    
    structure = {
        'file_path': file_path,
        'file_name': os.path.basename(file_path),
        'file_size_mb': os.path.getsize(file_path) / (1024 * 1024),
        'root_attributes': {},
        'devices': {},
        'other_groups': {},
        'metadata': {}
    }
    
    try:
        with h5py.File(file_path, 'r') as f:
            # Extract root attributes
            structure['root_attributes'] = extract_attributes(f)
            
            # Check for each group at root level
            for group_name in f:
                group = f[group_name]
                
                if isinstance(group, h5py.Group):
                    group_info = {
                        'attributes': extract_attributes(group),
                        'children': {}
                    }
                    
                    # Special handling for devices group
                    if group_name == 'devices':
                        # Process each device
                        for device_name in group:
                            device_group = group[device_name]
                            if isinstance(device_group, h5py.Group):
                                device_info = {
                                    'attributes': extract_attributes(device_group),
                                    'datasets': {},
                                    'subgroups': {}
                                }
                                
                                # Process each item in the device group
                                for item_name in device_group:
                                    item = device_group[item_name]
                                    if isinstance(item, h5py.Dataset):
                                        # It's a dataset
                                        dataset_info = {
                                            'shape': item.shape,
                                            'dtype': str(item.dtype),
                                            'size_mb': item.size * item.dtype.itemsize / (1024 * 1024),
                                            'attributes': extract_attributes(item)
                                        }
                                        device_info['datasets'][item_name] = dataset_info
                                        
                                        # For timestamps dataset, get extra info
                                        if item_name == 'timestamps' and item.size > 0:
                                            try:
                                                timestamps = item[:]
                                                if len(timestamps) > 1:
                                                    # Get duration using first column if multi-column
                                                    first_ts = timestamps[0][0] if timestamps.ndim > 1 else timestamps[0]
                                                    last_ts = timestamps[-1][0] if timestamps.ndim > 1 else timestamps[-1]
                                                    duration_ms = last_ts - first_ts
                                                    duration_sec = duration_ms / 1000
                                                    
                                                    # Calculate sample rate from time differences
                                                    if timestamps.ndim > 1:
                                                        time_diffs = np.diff(timestamps[:min(100, len(timestamps))], axis=0)[:,0]
                                                    else:
                                                        time_diffs = np.diff(timestamps[:min(100, len(timestamps))])
                                                    
                                                    mean_diff_ms = np.mean(time_diffs)
                                                    sample_rate = 1000 / mean_diff_ms if mean_diff_ms > 0 else 0
                                                    
                                                    dataset_info['first_timestamp'] = float(first_ts)
                                                    dataset_info['last_timestamp'] = float(last_ts)
                                                    dataset_info['duration_ms'] = float(duration_ms)
                                                    dataset_info['duration_sec'] = float(duration_sec)
                                                    dataset_info['mean_sample_interval_ms'] = float(mean_diff_ms)
                                                    dataset_info['estimated_sample_rate_hz'] = float(sample_rate)
                                            except Exception as e:
                                                logger.warning(f"Error processing timestamps: {e}")
                                    elif isinstance(item, h5py.Group):
                                        # It's a subgroup
                                        subgroup_info = {
                                            'attributes': extract_attributes(item),
                                            'children': {}
                                        }
                                        
                                        # Process items in subgroup
                                        for subitem_name in item:
                                            subitem = item[subitem_name]
                                            if isinstance(subitem, h5py.Dataset):
                                                # Add basic info about dataset
                                                subgroup_info['children'][subitem_name] = {
                                                    'type': 'dataset',
                                                    'shape': subitem.shape,
                                                    'dtype': str(subitem.dtype),
                                                    'attributes': extract_attributes(subitem)
                                                }
                                            elif isinstance(subitem, h5py.Group):
                                                # Note that it's a group
                                                subgroup_info['children'][subitem_name] = {
                                                    'type': 'group',
                                                    'attributes': extract_attributes(subitem)
                                                }
                                        
                                        device_info['subgroups'][item_name] = subgroup_info
                                
                                structure['devices'][device_name] = device_info
                        
                    # Special handling for metadata group
                    elif group_name == 'metadata':
                        # Extract all metadata values
                        for meta_name in group:
                            item = group[meta_name]
                            if isinstance(item, h5py.Dataset):
                                try:
                                    # Read the dataset value
                                    value = item[()]
                                    # Convert to Python types
                                    if isinstance(value, np.ndarray):
                                        value = value.tolist()
                                    elif isinstance(value, (np.int32, np.int64)):
                                        value = int(value)
                                    elif isinstance(value, (np.float32, np.float64)):
                                        value = float(value)
                                    elif isinstance(value, np.bool_):
                                        value = bool(value)
                                    elif isinstance(value, bytes):
                                        try:
                                            value = value.decode('utf-8')
                                        except:
                                            value = str(value)
                                    
                                    structure['metadata'][meta_name] = value
                                except Exception as e:
                                    structure['metadata'][meta_name] = f"[Error reading value: {str(e)}]"
                    else:
                        # For other groups, just store attributes and note that they exist
                        structure['other_groups'][group_name] = group_info
    
    except Exception as e:
        logger.error(f"Error extracting structure: {e}", exc_info=True)
    
    return structure


def download_and_extract_structure(bucket, key):
    """Download an H5 file from S3 and extract its structure.
    
    Args:
        bucket: S3 bucket name
        key: S3 object key
        
    Returns:
        Dictionary with structure information
    """
    # Add parent directory to path to import utils
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
    from utils.aws import init_s3_client
    
    logger.info(f"Downloading and extracting structure from s3://{bucket}/{key}")
    
    # Create temporary directory for download
    temp_dir = tempfile.mkdtemp(prefix="device_metadata_")
    local_path = os.path.join(temp_dir, os.path.basename(key))
    
    try:
        # Initialize S3 client
        s3_client = init_s3_client()
        
        # Download the file
        logger.info(f"Downloading to {local_path}")
        s3_client.download_file(bucket, key, local_path)
        logger.info(f"Downloaded {os.path.getsize(local_path)/1024/1024:.2f} MB")
        
        # Extract structure
        structure = extract_h5_structure(local_path)
        
        # Clean up temporary file
        os.remove(local_path)
        os.rmdir(temp_dir)
        
        return structure
    
    except Exception as e:
        logger.error(f"Error downloading or extracting structure: {e}", exc_info=True)
        return {
            'error': str(e),
            'file_path': f"s3://{bucket}/{key}"
        }


def print_structure_summary(structure):
    """Print a summary of the structure and metadata.
    
    Args:
        structure: Dictionary with structure information
    """
    print("\n" + "="*80)
    print(f"H5 FILE STRUCTURE AND METADATA SUMMARY")
    print("="*80)
    
    if 'error' in structure:
        print(f"Error: {structure['error']}")
        return
    
    print(f"File: {structure['file_name']}")
    print(f"Size: {structure['file_size_mb']:.2f} MB")
    
    # Print root attributes if any
    if structure['root_attributes']:
        print("\nRoot Attributes:")
        for key, value in structure['root_attributes'].items():
            print(f"  {key}: {value}")
    
    # Print metadata if any
    if structure['metadata']:
        print("\nMetadata:")
        for key, value in structure['metadata'].items():
            print(f"  {key}: {value}")
    
    # Print device information
    if structure['devices']:
        print("\nDevices:")
        for device_name, device_info in structure['devices'].items():
            print(f"\n  {device_name.upper()}:")
            
            # Print device attributes
            if device_info['attributes']:
                print("    Attributes:")
                for key, value in device_info['attributes'].items():
                    print(f"      {key}: {value}")
            
            # Print device datasets
            if device_info['datasets']:
                print("    Datasets:")
                for dataset_name, dataset_info in device_info['datasets'].items():
                    print(f"      {dataset_name}:")
                    print(f"        Shape: {dataset_info['shape']}")
                    print(f"        Size: {dataset_info['size_mb']:.2f} MB")
                    print(f"        Type: {dataset_info['dtype']}")
                    
                    # Print timestamps-specific info if available
                    if 'duration_sec' in dataset_info:
                        print(f"        Time Range: {dataset_info['first_timestamp']} to {dataset_info['last_timestamp']} ms")
                        print(f"        Duration: {dataset_info['duration_sec']:.2f} seconds ({dataset_info['duration_sec']/60:.2f} minutes)")
                        print(f"        Sample Rate: {dataset_info['estimated_sample_rate_hz']:.2f} Hz")
                    
                    # Print dataset attributes if any
                    if dataset_info['attributes']:
                        print("        Attributes:")
                        for key, value in dataset_info['attributes'].items():
                            print(f"          {key}: {value}")
            
            # Print device subgroups
            if device_info['subgroups']:
                print("    Subgroups:")
                for subgroup_name, subgroup_info in device_info['subgroups'].items():
                    print(f"      {subgroup_name}:")
                    
                    # Print subgroup attributes
                    if subgroup_info['attributes']:
                        print("        Attributes:")
                        for key, value in subgroup_info['attributes'].items():
                            print(f"          {key}: {value}")
                    
                    # Summarize subgroup children
                    if subgroup_info['children']:
                        print("        Children:")
                        for child_name, child_info in subgroup_info['children'].items():
                            if child_info['type'] == 'dataset':
                                print(f"          {child_name}: Dataset {child_info['shape']} ({child_info['dtype']})")
                            else:
                                print(f"          {child_name}: Group")
    
    # Summarize other groups
    if structure['other_groups']:
        print("\nOther Groups:")
        for group_name, group_info in structure['other_groups'].items():
            print(f"  {group_name}:")
            
            # Print group attributes
            if group_info['attributes']:
                print("    Attributes:")
                for key, value in group_info['attributes'].items():
                    print(f"      {key}: {value}")
            
            # Summarize child count
            print(f"    Children: {len(group_info['children'])} items")
    
    print("="*80)


def main():
    """Main function to run the script."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Extract and display device metadata from H5 files")
    parser.add_argument("--bucket", default="conduit-data-dev", help="S3 bucket name")
    parser.add_argument("--key", required=True, help="S3 object key")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument("--json", action="store_true", help="Output in JSON format")
    parser.add_argument("--output", help="Output file path for JSON format")
    args = parser.parse_args()
    
    # Set log level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Download and extract structure
    structure = download_and_extract_structure(args.bucket, args.key)
    
    # Output results
    if args.json:
        # Output in JSON format
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(structure, f, indent=2)
            print(f"Structure saved to {args.output}")
        else:
            print(json.dumps(structure, indent=2))
    else:
        # Print human-readable summary
        print_structure_summary(structure)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())