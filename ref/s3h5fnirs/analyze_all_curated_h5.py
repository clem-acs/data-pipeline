#!/usr/bin/env python3
"""
Script to analyze all curated H5 files in the S3 bucket.

This script:
1. Lists all H5 files in the curated-h5/ S3 prefix
2. For each file, analyzes the fNIRS data structure
3. Reports statistics on -inf values and valid channels
4. Generates a summary report

Usage:
python analyze_all_curated_h5.py [--limit N] [--sample-frames N]
"""

import os
import sys
import argparse
import logging
import tempfile
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("curated_h5_analysis")


def list_curated_h5_files(s3_client, bucket, prefix='curated-h5/', limit=None):
    """List curated H5 files from S3.
    
    Args:
        s3_client: Boto3 S3 client
        bucket: S3 bucket name
        prefix: S3 prefix to search
        limit: Maximum number of files to list
        
    Returns:
        List of dictionaries with file info (key, size, etc.)
    """
    logger.info(f"Listing H5 files from s3://{bucket}/{prefix}")
    
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
                
                if limit and len(files) >= limit:
                    break
        
        if limit and len(files) >= limit:
            break
    
    logger.info(f"Found {len(files)} H5 files")
    return files


def analyze_h5_file(file_path, sample_frames=5):
    """Analyze a single H5 file for fNIRS data and -inf values.
    
    Args:
        file_path: Path to the H5 file
        sample_frames: Number of frames to sample
        
    Returns:
        Dictionary with analysis results
    """
    logger.info(f"Analyzing H5 file: {file_path}")
    
    results = {
        'file_path': file_path,
        'file_name': os.path.basename(file_path),
        'file_size_mb': os.path.getsize(file_path) / (1024 * 1024),
        'dataset_path': None,
        'dataset_shape': None,
        'frames': 0,
        'channels': 0,
        'total_elements': 0,
        'inf_channels': 0,
        'valid_channels': 0,
        'inf_percentage': 0,
        'valid_percentage': 0,
        'error': None
    }
    
    try:
        with h5py.File(file_path, 'r') as f:
            # Check if "devices/fnirs/frames_data" exists
            if 'devices/fnirs/frames_data' in f:
                dataset_path = 'devices/fnirs/frames_data'
                dataset = f[dataset_path]
                
                # Record basic dataset information
                results['dataset_path'] = dataset_path
                results['dataset_shape'] = dataset.shape
                results['frames'] = dataset.shape[0]
                results['channels'] = dataset.shape[1]
                results['total_elements'] = dataset.shape[0] * dataset.shape[1]
                
                logger.info(f"Found fNIRS dataset at {dataset_path} with shape {dataset.shape}")
                
                # Sample random frames
                if sample_frames >= dataset.shape[0]:
                    # Analyze all frames
                    frame_indices = np.arange(dataset.shape[0])
                else:
                    # Sample random frames
                    frame_indices = np.random.choice(dataset.shape[0], size=sample_frames, replace=False)
                    frame_indices.sort()  # Sort to ensure sequential reading
                
                # Track channel status over all sampled frames
                frames_analyzed = len(frame_indices)
                inf_channel_count = np.zeros(dataset.shape[1], dtype=bool)
                
                # Analyze each sampled frame
                for frame_idx in frame_indices:
                    logger.debug(f"Processing frame {frame_idx + 1}/{len(frame_indices)}")
                    
                    # Read frame
                    frame = dataset[frame_idx]
                    
                    # Reshape if needed
                    if len(frame.shape) > 1 and frame.shape[-1] == 1:
                        frame = frame.reshape(frame.shape[0])
                    
                    # Mark channels with -inf as True in our tracking array
                    for channel_idx in range(dataset.shape[1]):
                        # Get channel value
                        channel_value = frame[channel_idx]
                        
                        # Check if scalar or array
                        if isinstance(channel_value, np.ndarray):
                            if np.any(np.isneginf(channel_value)):
                                inf_channel_count[channel_idx] = True
                        else:
                            if np.isneginf(channel_value):
                                inf_channel_count[channel_idx] = True
                
                # Calculate total inf and valid channels
                inf_channels = np.sum(inf_channel_count)
                valid_channels = dataset.shape[1] - inf_channels
                
                results['inf_channels'] = int(inf_channels)
                results['valid_channels'] = int(valid_channels)
                results['inf_percentage'] = (inf_channels / dataset.shape[1]) * 100
                results['valid_percentage'] = (valid_channels / dataset.shape[1]) * 100
            else:
                logger.warning(f"No 'devices/fnirs/frames_data' found in {file_path}")
                results['error'] = "No fNIRS dataset found"
    
    except Exception as e:
        logger.error(f"Error analyzing H5 file: {e}", exc_info=True)
        results['error'] = str(e)
    
    return results


def download_and_analyze_s3_h5(s3_client, bucket, key, sample_frames=5):
    """Download an H5 file from S3 and analyze it.
    
    Args:
        s3_client: Boto3 S3 client
        bucket: S3 bucket name
        key: S3 object key
        sample_frames: Number of frames to sample
        
    Returns:
        Dictionary with analysis results
    """
    logger.info(f"Downloading and analyzing s3://{bucket}/{key}")
    
    # Create temporary directory for download
    temp_dir = tempfile.mkdtemp(prefix="h5_analysis_")
    local_path = os.path.join(temp_dir, os.path.basename(key))
    
    try:
        # Download the file
        logger.info(f"Downloading to {local_path}")
        s3_client.download_file(bucket, key, local_path)
        logger.info(f"Downloaded {os.path.getsize(local_path)/1024/1024:.2f} MB")
        
        # Analyze the file
        results = analyze_h5_file(local_path, sample_frames)
        
        # Add S3 info to results
        results['s3_bucket'] = bucket
        results['s3_key'] = key
        
        # Clean up temporary file
        os.remove(local_path)
        os.rmdir(temp_dir)
        
        return results
    
    except Exception as e:
        logger.error(f"Error downloading or analyzing file: {e}", exc_info=True)
        return {
            'file_name': os.path.basename(key),
            's3_bucket': bucket,
            's3_key': key,
            'error': str(e)
        }


def analyze_all_h5_files(bucket, prefix='curated-h5/', limit=None, sample_frames=5):
    """Analyze all H5 files in the specified S3 bucket and prefix.
    
    Args:
        bucket: S3 bucket name
        prefix: S3 prefix to search
        limit: Maximum number of files to analyze
        sample_frames: Number of frames to sample per file
        
    Returns:
        DataFrame with analysis results
    """
    # Add parent directory to path to import utils
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
    from utils.aws import init_s3_client
    
    # Initialize S3 client
    s3_client = init_s3_client()
    
    # List H5 files
    h5_files = list_curated_h5_files(s3_client, bucket, prefix, limit)
    
    if not h5_files:
        logger.warning(f"No H5 files found in s3://{bucket}/{prefix}")
        return pd.DataFrame()
    
    # Analyze each file
    results = []
    
    for i, file_info in enumerate(h5_files):
        logger.info(f"Analyzing file {i+1}/{len(h5_files)}: {file_info['key']}")
        
        # Download and analyze
        file_results = download_and_analyze_s3_h5(s3_client, bucket, file_info['key'], sample_frames)
        
        # Add to results
        results.append(file_results)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    return df


def generate_report(df, output_dir='.'):
    """Generate a report from the analysis results.
    
    Args:
        df: DataFrame with analysis results
        output_dir: Directory to save the report
        
    Returns:
        Path to the report file
    """
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Define report path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(output_dir, f"curated_h5_analysis_{timestamp}.csv")
    
    # Calculate summary statistics
    total_files = len(df)
    files_with_fnirs = df['dataset_path'].notna().sum()
    
    # Write report to CSV
    df.to_csv(report_path, index=False)
    
    # Print summary
    print("\n" + "="*80)
    print(f"CURATED H5 FILES ANALYSIS SUMMARY")
    print("="*80)
    print(f"Total files analyzed: {total_files}")
    print(f"Files with fNIRS data: {files_with_fnirs} ({files_with_fnirs/total_files*100:.2f}%)")
    
    if files_with_fnirs > 0:
        # Calculate average values for files with fNIRS data
        fnirs_df = df[df['dataset_path'].notna()]
        
        avg_channels = fnirs_df['channels'].mean()
        avg_valid_channels = fnirs_df['valid_channels'].mean()
        avg_valid_percentage = fnirs_df['valid_percentage'].mean()
        
        print(f"\nAverage total channels per file: {avg_channels:.2f}")
        print(f"Average valid channels per file: {avg_valid_channels:.2f} ({avg_valid_percentage:.2f}%)")
        print(f"Average channels with -inf values: {avg_channels - avg_valid_channels:.2f} ({100-avg_valid_percentage:.2f}%)")
        
        # Show file with most and least valid channels
        most_valid = fnirs_df.loc[fnirs_df['valid_percentage'].idxmax()]
        least_valid = fnirs_df.loc[fnirs_df['valid_percentage'].idxmin()]
        
        print(f"\nFile with most valid channels: {most_valid['file_name']}")
        print(f"  Valid channels: {most_valid['valid_channels']} of {most_valid['channels']} ({most_valid['valid_percentage']:.2f}%)")
        
        print(f"\nFile with least valid channels: {least_valid['file_name']}")
        print(f"  Valid channels: {least_valid['valid_channels']} of {least_valid['channels']} ({least_valid['valid_percentage']:.2f}%)")
    
    print("\nDetailed results saved to:")
    print(f"  {report_path}")
    print("="*80)
    
    return report_path


def main():
    """Main function to run the script."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Analyze all curated H5 files")
    parser.add_argument("--bucket", default="conduit-data-dev", help="S3 bucket name")
    parser.add_argument("--prefix", default="curated-h5/", help="S3 prefix for curated H5 files")
    parser.add_argument("--limit", type=int, help="Maximum number of files to analyze")
    parser.add_argument("--sample-frames", type=int, default=5, help="Number of frames to sample per file")
    parser.add_argument("--output-dir", default=".", help="Directory to save the report")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()
    
    # Set log level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Analyze all H5 files
    results = analyze_all_h5_files(args.bucket, args.prefix, args.limit, args.sample_frames)
    
    # Generate report
    if not results.empty:
        generate_report(results, args.output_dir)
    else:
        logger.error("No results to report")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())