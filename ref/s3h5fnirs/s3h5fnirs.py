#!/usr/bin/env python3
"""
Script to download and analyze fNIRS data from curated H5 files in S3.

This script:
1. Lists H5 files from the curated-h5/ S3 prefix
2. Downloads a sample H5 file
3. Explores the fNIRS data within the H5 file
4. Outputs information about the data structure and shapes

Usage:
python s3h5fnirs.py [--sample-count N] [--download-dir DIR]
"""

import os
import sys
import argparse
import tempfile
import logging
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path to import utils
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from utils.aws import init_s3_client, get_aws_credentials

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("s3h5fnirs")


def list_curated_h5_files(s3_client, bucket, prefix='curated-h5/', max_files=10):
    """List curated H5 files from S3.
    
    Args:
        s3_client: Boto3 S3 client
        bucket: S3 bucket name
        prefix: S3 prefix to search
        max_files: Maximum number of files to list
        
    Returns:
        List of dictionaries with file info (key, size, etc.)
    """
    logger.info(f"Listing up to {max_files} H5 files from s3://{bucket}/{prefix}")
    
    paginator = s3_client.get_paginator('list_objects_v2')
    files = []
    
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        if 'Contents' in page:
            for obj in page['Contents']:
                if obj['Key'].endswith('.h5'):
                    files.append({
                        'key': obj['Key'],
                        'size': obj['Size'],
                        'last_modified': obj['LastModified']
                    })
                
                if len(files) >= max_files:
                    break
        
        if len(files) >= max_files:
            break
    
    logger.info(f"Found {len(files)} H5 files")
    return files


def download_h5_file(s3_client, bucket, file_info, download_dir=None):
    """Download an H5 file from S3.
    
    Args:
        s3_client: Boto3 S3 client
        bucket: S3 bucket name
        file_info: Dictionary with file information
        download_dir: Directory to download to (uses temp dir if None)
        
    Returns:
        Path to downloaded file
    """
    # Use download_dir if provided, otherwise create temp dir
    if download_dir:
        os.makedirs(download_dir, exist_ok=True)
        local_path = os.path.join(download_dir, os.path.basename(file_info['key']))
    else:
        temp_dir = tempfile.mkdtemp(prefix="s3h5fnirs_")
        local_path = os.path.join(temp_dir, os.path.basename(file_info['key']))
    
    logger.info(f"Downloading s3://{bucket}/{file_info['key']} to {local_path}")
    
    # Download the file
    s3_client.download_file(bucket, file_info['key'], local_path)
    
    logger.info(f"Downloaded {os.path.getsize(local_path)/1024/1024:.2f} MB")
    return local_path


def explore_h5_file_structure(h5_file_path):
    """Explore the structure of an H5 file.
    
    Args:
        h5_file_path: Path to H5 file
        
    Returns:
        Dictionary with file structure information
    """
    logger.info(f"Exploring structure of {h5_file_path}")
    
    structure = {
        'groups': [],
        'datasets': []
    }
    
    def explore_group(name, obj):
        """Callback to explore H5 groups and datasets."""
        if isinstance(obj, h5py.Group):
            structure['groups'].append({
                'name': name,
                'attrs': {key: obj.attrs[key] for key in obj.attrs}
            })
        elif isinstance(obj, h5py.Dataset):
            structure['datasets'].append({
                'name': name,
                'shape': obj.shape,
                'dtype': str(obj.dtype),
                'attrs': {key: obj.attrs[key] for key in obj.attrs},
                'size_mb': obj.size * obj.dtype.itemsize / (1024 * 1024)
            })
    
    with h5py.File(h5_file_path, 'r') as f:
        # Visit all groups and datasets
        f.visititems(explore_group)
    
    return structure


def find_fnirs_data(h5_file_path):
    """Find and analyze fNIRS data in an H5 file.
    
    Args:
        h5_file_path: Path to H5 file
        
    Returns:
        Dictionary with fNIRS data information
    """
    logger.info(f"Looking for fNIRS data in {h5_file_path}")
    
    fnirs_info = {
        'found': False,
        'data_paths': [],
        'timestamps_paths': [],
        'data_shapes': [],
        'sample_rates': [],
        'sample_data': []
    }
    
    with h5py.File(h5_file_path, 'r') as f:
        # Look for common fNIRS-related group names
        potential_fnirs_paths = []
        
        # Find all potential fNIRS datasets or groups
        for device_pattern in ['devices/', 'fnirs/', 'nirs/', 'NIRx/', 'fNIRS/', 'TD/', 'fd/']:
            if device_pattern in f:
                logger.info(f"Found device path: {device_pattern}")
                potential_fnirs_paths.append(device_pattern)
        
        # Search more generally for any keys containing 'nir' case-insensitive
        for key in f.keys():
            key_lower = key.lower()
            if 'nir' in key_lower and key not in potential_fnirs_paths:
                logger.info(f"Found potential fNIRS key: {key}")
                potential_fnirs_paths.append(key)
        
        # If no obvious fNIRS paths found, look at all datasets
        if not potential_fnirs_paths:
            logger.info("No obvious fNIRS paths found, checking all datasets")
            
            def collect_datasets(name, obj):
                if isinstance(obj, h5py.Dataset) and len(obj.shape) >= 2:
                    # Look for time series data (typically has a long first dimension)
                    if obj.shape[0] > 1000:
                        potential_fnirs_paths.append(name)
            
            f.visititems(collect_datasets)
        
        # Analyze each potential fNIRS path
        for path in potential_fnirs_paths:
            logger.info(f"Analyzing potential fNIRS path: {path}")
            
            # Check if it's a dataset or group
            if isinstance(f[path], h5py.Dataset):
                # It's a dataset, check if it looks like time series data
                dataset = f[path]
                if len(dataset.shape) >= 2 and dataset.shape[0] > 1000:
                    fnirs_info['found'] = True
                    fnirs_info['data_paths'].append(path)
                    fnirs_info['data_shapes'].append(dataset.shape)
                    
                    # Get sample data
                    sample_rows = min(5, dataset.shape[0])
                    sample_cols = min(5, dataset.shape[1] if len(dataset.shape) > 1 else 1)
                    
                    if len(dataset.shape) == 1:
                        sample = dataset[:sample_rows]
                    else:
                        sample = dataset[:sample_rows, :sample_cols]
                    
                    fnirs_info['sample_data'].append({
                        'path': path,
                        'data': sample
                    })
                    
                    # Look for associated timestamps
                    parent_group = path.rsplit('/', 1)[0] if '/' in path else ''
                    for ts_name in ['timestamps', 'time', 'timeStamps', 'timepoints']:
                        ts_path = f"{parent_group}/{ts_name}" if parent_group else ts_name
                        if ts_path in f:
                            fnirs_info['timestamps_paths'].append(ts_path)
                            
                            # Calculate sample rate if timestamps available
                            timestamps = f[ts_path]
                            if len(timestamps) > 1:
                                # Try to calculate sample rate from first few timestamps
                                sample_indices = min(100, len(timestamps)-1)
                                time_diffs = np.diff(timestamps[:sample_indices])
                                mean_diff = np.mean(time_diffs)
                                
                                if mean_diff > 0:
                                    # Convert to Hz depending on timestamp units
                                    # Assume timestamps are in seconds if < 1000, otherwise milliseconds
                                    if mean_diff < 1000:
                                        sample_rate = 1.0 / mean_diff  # seconds to Hz
                                    else:
                                        sample_rate = 1000.0 / mean_diff  # ms to Hz
                                    
                                    fnirs_info['sample_rates'].append(sample_rate)
                                else:
                                    fnirs_info['sample_rates'].append(None)
            else:
                # It's a group, search for datasets within it
                group = f[path]
                
                # Check if this group or its children have datasets that look like fNIRS data
                for child_name in group:
                    child_path = f"{path}/{child_name}"
                    child = group[child_name]
                    
                    if isinstance(child, h5py.Dataset) and len(child.shape) >= 2 and child.shape[0] > 1000:
                        fnirs_info['found'] = True
                        fnirs_info['data_paths'].append(child_path)
                        fnirs_info['data_shapes'].append(child.shape)
                        
                        # Get sample data
                        sample_rows = min(5, child.shape[0])
                        sample_cols = min(5, child.shape[1] if len(child.shape) > 1 else 1)
                        
                        if len(child.shape) == 1:
                            sample = child[:sample_rows]
                        else:
                            sample = child[:sample_rows, :sample_cols]
                        
                        fnirs_info['sample_data'].append({
                            'path': child_path,
                            'data': sample
                        })
                        
                        # Look for associated timestamps in the same group
                        for ts_name in ['timestamps', 'time', 'timeStamps', 'timepoints']:
                            ts_path = f"{path}/{ts_name}"
                            if ts_path in f:
                                fnirs_info['timestamps_paths'].append(ts_path)
                                
                                # Calculate sample rate if timestamps available
                                timestamps = f[ts_path]
                                if len(timestamps) > 1:
                                    # Try to calculate sample rate from first few timestamps
                                    sample_indices = min(100, len(timestamps)-1)
                                    time_diffs = np.diff(timestamps[:sample_indices])
                                    mean_diff = np.mean(time_diffs)
                                    
                                    if mean_diff > 0:
                                        # Convert to Hz depending on timestamp units
                                        # Assume timestamps are in seconds if < 1000, otherwise milliseconds
                                        if mean_diff < 1000:
                                            sample_rate = 1.0 / mean_diff  # seconds to Hz
                                        else:
                                            sample_rate = 1000.0 / mean_diff  # ms to Hz
                                        
                                        fnirs_info['sample_rates'].append(sample_rate)
                                    else:
                                        fnirs_info['sample_rates'].append(None)
    
    return fnirs_info


def visualize_sample_data(fnirs_info, output_dir=None):
    """Visualize sample fNIRS data.
    
    Args:
        fnirs_info: Dictionary with fNIRS data information
        output_dir: Directory to save visualizations (uses current dir if None)
    
    Returns:
        List of paths to generated visualizations
    """
    # Skip if no fNIRS data found
    if not fnirs_info['found'] or not fnirs_info['sample_data']:
        logger.warning("No fNIRS data to visualize")
        return []
    
    # Use output_dir if provided, otherwise use current directory
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = os.getcwd()
    
    visualization_paths = []
    
    # Visualize each sample dataset
    for i, sample_info in enumerate(fnirs_info['sample_data']):
        path = sample_info['path']
        data = sample_info['data']
        
        # Create a simple plot of the first few channels
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # If 1D data, plot as is
        if len(data.shape) == 1:
            ax.plot(data)
            ax.set_title(f"Sample data from {path}")
            ax.set_xlabel("Time points")
            ax.set_ylabel("Value")
        # If 2D data, plot first few channels
        elif len(data.shape) == 2:
            for ch in range(min(5, data.shape[1])):
                ax.plot(data[:, ch], label=f"Channel {ch}")
            ax.set_title(f"Sample data from {path}")
            ax.set_xlabel("Time points")
            ax.set_ylabel("Value")
            ax.legend()
        
        # Save figure
        safe_path = path.replace('/', '_').replace('\\', '_')
        output_path = os.path.join(output_dir, f"fnirs_sample_{i}_{safe_path}.png")
        fig.savefig(output_path)
        plt.close(fig)
        
        visualization_paths.append(output_path)
        logger.info(f"Saved visualization to {output_path}")
    
    return visualization_paths


def create_report(h5_file_path, structure, fnirs_info, visualizations=None):
    """Create a report of fNIRS data analysis.
    
    Args:
        h5_file_path: Path to H5 file
        structure: Dictionary with file structure information
        fnirs_info: Dictionary with fNIRS data information
        visualizations: List of paths to visualizations
        
    Returns:
        Path to report file
    """
    report_dir = os.path.dirname(h5_file_path)
    report_path = os.path.join(report_dir, "fnirs_analysis_report.md")
    
    with open(report_path, 'w') as f:
        # Write header
        f.write(f"# fNIRS Data Analysis Report\n\n")
        f.write(f"File: `{os.path.basename(h5_file_path)}`\n\n")
        
        # Write summary
        f.write("## Summary\n\n")
        f.write(f"- **fNIRS data found:** {'Yes' if fnirs_info['found'] else 'No'}\n")
        if fnirs_info['found']:
            f.write(f"- **Number of data paths:** {len(fnirs_info['data_paths'])}\n")
            
            # Write data shapes
            f.write("- **Data shapes:**\n")
            for i, (path, shape) in enumerate(zip(fnirs_info['data_paths'], fnirs_info['data_shapes'])):
                f.write(f"  - `{path}`: {shape}\n")
            
            # Write sample rates if available
            if fnirs_info['sample_rates']:
                f.write("- **Sample rates:**\n")
                for i, (path, rate) in enumerate(zip(fnirs_info['data_paths'], fnirs_info['sample_rates'])):
                    if rate is not None:
                        f.write(f"  - `{path}`: {rate:.2f} Hz\n")
                    else:
                        f.write(f"  - `{path}`: Unknown\n")
        
        # Write file structure
        f.write("\n## File Structure\n\n")
        f.write("### Groups\n\n")
        for group in structure['groups']:
            f.write(f"- `{group['name']}`\n")
            if group['attrs']:
                f.write("  - Attributes:\n")
                for key, value in group['attrs'].items():
                    f.write(f"    - `{key}`: {value}\n")
        
        f.write("\n### Datasets\n\n")
        for dataset in sorted(structure['datasets'], key=lambda x: x['size_mb'], reverse=True):
            f.write(f"- `{dataset['name']}`\n")
            f.write(f"  - Shape: {dataset['shape']}\n")
            f.write(f"  - Size: {dataset['size_mb']:.2f} MB\n")
            f.write(f"  - Type: {dataset['dtype']}\n")
            if dataset['attrs']:
                f.write("  - Attributes:\n")
                for key, value in dataset['attrs'].items():
                    f.write(f"    - `{key}`: {value}\n")
        
        # Include links to visualizations if available
        if visualizations:
            f.write("\n## Visualizations\n\n")
            for vis_path in visualizations:
                vis_name = os.path.basename(vis_path)
                f.write(f"- ![{vis_name}]({vis_name})\n")
    
    logger.info(f"Created report at {report_path}")
    return report_path


def main():
    """Main function to run the script."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Download and analyze fNIRS data from curated H5 files")
    parser.add_argument('--bucket', default='conduit-data-dev', help='S3 bucket name')
    parser.add_argument('--prefix', default='curated-h5/', help='S3 prefix for curated H5 files')
    parser.add_argument('--sample-count', type=int, default=1, help='Number of H5 files to sample')
    parser.add_argument('--download-dir', help='Directory to download H5 files to')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    args = parser.parse_args()
    
    # Set log level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    try:
        # Initialize S3 client
        s3_client = init_s3_client()
        
        # List curated H5 files
        h5_files = list_curated_h5_files(
            s3_client, 
            args.bucket, 
            args.prefix, 
            max_files=args.sample_count
        )
        
        if not h5_files:
            logger.error(f"No H5 files found in s3://{args.bucket}/{args.prefix}")
            return 1
        
        # Analyze each file
        for file_info in h5_files:
            # Download the file
            h5_file_path = download_h5_file(
                s3_client, 
                args.bucket, 
                file_info, 
                args.download_dir
            )
            
            # Explore the file structure
            structure = explore_h5_file_structure(h5_file_path)
            
            # Find and analyze fNIRS data
            fnirs_info = find_fnirs_data(h5_file_path)
            
            # Visualize sample data
            visualizations = visualize_sample_data(
                fnirs_info, 
                os.path.dirname(h5_file_path)
            )
            
            # Create report
            report_path = create_report(
                h5_file_path, 
                structure, 
                fnirs_info, 
                visualizations
            )
            
            # Print report path
            logger.info(f"Analysis complete. Report: {report_path}")
            
            # Print summary to console
            print("\n" + "="*80)
            print(f"ANALYSIS SUMMARY FOR {os.path.basename(h5_file_path)}")
            print("="*80)
            print(f"fNIRS data found: {'Yes' if fnirs_info['found'] else 'No'}")
            
            if fnirs_info['found']:
                print(f"Number of data paths: {len(fnirs_info['data_paths'])}")
                
                print("Data paths and shapes:")
                for path, shape in zip(fnirs_info['data_paths'], fnirs_info['data_shapes']):
                    print(f"  - {path}: {shape}")
                
                if fnirs_info['timestamps_paths']:
                    print("Timestamp paths:")
                    for path in fnirs_info['timestamps_paths']:
                        print(f"  - {path}")
                
                if fnirs_info['sample_rates']:
                    print("Sample rates:")
                    for path, rate in zip(fnirs_info['data_paths'], fnirs_info['sample_rates']):
                        if rate is not None:
                            print(f"  - {path}: {rate:.2f} Hz")
                        else:
                            print(f"  - {path}: Unknown")
            
            # Compare with SNIRF format
            print("\nComparison with SNIRF format:")
            if fnirs_info['found']:
                print("This H5 file contains fNIRS data but in a custom format, not SNIRF.")
                print("SNIRF files would have standard structures like:")
                print("  - /nirs/data1/dataTimeSeries (main data)")
                print("  - /nirs/data1/measurementList (channel info)")
                print("  - /nirs/probe (optode positions)")
            else:
                print("No fNIRS data found in a recognizable format.")
            
            print("\nReport and visualizations saved to:")
            print(f"  - {report_path}")
            for vis in visualizations:
                print(f"  - {vis}")
            print("="*80 + "\n")
        
        return 0
    
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=args.verbose)
        return 1


if __name__ == "__main__":
    sys.exit(main())