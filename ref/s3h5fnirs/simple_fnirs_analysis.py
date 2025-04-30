#!/usr/bin/env python3
"""
Simple script to analyze fNIRS data in H5 files for NaN, zero, and non-zero values.

This script provides a straightforward analysis of the fNIRS data values
in a curated H5 file, reporting the percentage of NaN, zero, and non-zero values.

Usage:
python simple_fnirs_analysis.py --s3 --key "curated-h5/filename.h5" --chunk-size 1000
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
logger = logging.getLogger("fnirs_analysis")


def analyze_fnirs_in_h5(file_path, chunk_size=1000):
    """Analyze fNIRS data in an H5 file.
    
    Args:
        file_path: Path to the H5 file
        chunk_size: Number of frames to process at a time
        
    Returns:
        Dictionary with analysis results
    """
    logger.info(f"Analyzing fNIRS data in {file_path}")
    
    results = {
        'file_path': file_path,
        'dataset_path': None,
        'dataset_shape': None,
        'dataset_size_gb': None,
        'total_elements': 0,
        'nan_count': 0,
        'zero_count': 0,
        'non_zero_count': 0,
        'nan_percentage': None,
        'zero_percentage': None,
        'non_zero_percentage': None,
        'stats': None
    }
    
    try:
        with h5py.File(file_path, 'r') as f:
            # Check if "devices/fnirs/frames_data" exists
            if 'devices/fnirs/frames_data' in f:
                dataset_path = 'devices/fnirs/frames_data'
                dataset = f[dataset_path]
                
                # Record dataset information
                results['dataset_path'] = dataset_path
                results['dataset_shape'] = dataset.shape
                results['dataset_size_gb'] = dataset.size * dataset.dtype.itemsize / (1024**3)
                
                logger.info(f"Found fNIRS dataset at {dataset_path} with shape {dataset.shape}")
                logger.info(f"Dataset size: {results['dataset_size_gb']:.2f} GB")
                
                # Process the dataset in chunks along the first dimension
                total_elements = 0
                nan_count = 0
                zero_count = 0
                
                # Collect samples for statistics
                value_samples = []
                max_samples = 100000  # Maximum number of samples to collect
                sample_interval = max(1, dataset.shape[0] // (max_samples // 1000))
                
                # Process data in chunks
                for i in range(0, dataset.shape[0], chunk_size):
                    end_idx = min(i + chunk_size, dataset.shape[0])
                    logger.info(f"Processing frames {i} to {end_idx} of {dataset.shape[0]}")
                    
                    # Read chunk
                    chunk = dataset[i:end_idx]
                    
                    # Count values
                    chunk_nan_count = np.isnan(chunk).sum()
                    chunk_zero_count = np.count_nonzero(chunk == 0)
                    
                    # Update counts
                    total_elements += chunk.size
                    nan_count += chunk_nan_count
                    zero_count += chunk_zero_count
                    
                    # Collect samples at regular intervals
                    if i % sample_interval == 0:
                        # Get a small random sample from this chunk
                        sample_size = min(1000, chunk.size)
                        flat_chunk = chunk.flatten()
                        sample_indices = np.random.choice(flat_chunk.size, size=sample_size, replace=False)
                        samples = flat_chunk[sample_indices]
                        
                        # Filter out NaNs for value statistics
                        valid_samples = samples[~np.isnan(samples)]
                        if valid_samples.size > 0:
                            value_samples.append(valid_samples)
                
                # Calculate non-zero count
                non_zero_count = total_elements - nan_count - zero_count
                
                # Update results
                results['total_elements'] = total_elements
                results['nan_count'] = int(nan_count)
                results['zero_count'] = int(zero_count)
                results['non_zero_count'] = int(non_zero_count)
                
                # Calculate percentages
                results['nan_percentage'] = (nan_count / total_elements) * 100
                results['zero_percentage'] = (zero_count / total_elements) * 100
                results['non_zero_percentage'] = (non_zero_count / total_elements) * 100
                
                # Calculate statistics on collected samples
                if value_samples:
                    all_samples = np.concatenate(value_samples)
                    if all_samples.size > 0:
                        results['stats'] = {
                            'min': float(np.min(all_samples)),
                            'max': float(np.max(all_samples)),
                            'mean': float(np.mean(all_samples)),
                            'median': float(np.median(all_samples)),
                            'std': float(np.std(all_samples))
                        }
            else:
                logger.warning("No 'devices/fnirs/frames_data' found in the H5 file")
                
                # Look for any dataset with "fnirs" in the path
                fnirs_datasets = []
                
                def find_fnirs_datasets(name, obj):
                    if isinstance(obj, h5py.Dataset) and 'fnirs' in name.lower():
                        fnirs_datasets.append((name, obj))
                
                f.visititems(find_fnirs_datasets)
                
                if fnirs_datasets:
                    logger.info(f"Found {len(fnirs_datasets)} potential fNIRS datasets")
                    for name, dataset in fnirs_datasets:
                        logger.info(f"  {name}: {dataset.shape}, {dataset.dtype}")
                else:
                    logger.warning("No fNIRS datasets found in the file")
    
    except Exception as e:
        logger.error(f"Error analyzing H5 file: {e}", exc_info=True)
    
    return results


def download_and_analyze_s3_h5(bucket, key, chunk_size=1000):
    """Download an H5 file from S3 and analyze its fNIRS data.
    
    Args:
        bucket: S3 bucket name
        key: S3 object key
        chunk_size: Number of frames to process at a time
        
    Returns:
        Dictionary with analysis results
    """
    # Add parent directory to path to import utils
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
    from utils.aws import init_s3_client
    
    logger.info(f"Downloading and analyzing s3://{bucket}/{key}")
    
    # Create temporary directory for download
    temp_dir = tempfile.mkdtemp(prefix="fnirs_analysis_")
    local_path = os.path.join(temp_dir, os.path.basename(key))
    
    try:
        # Initialize S3 client
        s3_client = init_s3_client()
        
        # Download the file
        logger.info(f"Downloading to {local_path}")
        s3_client.download_file(bucket, key, local_path)
        logger.info(f"Downloaded {os.path.getsize(local_path)/1024/1024:.2f} MB")
        
        # Analyze the file
        results = analyze_fnirs_in_h5(local_path, chunk_size)
        
        # Clean up temporary file
        os.remove(local_path)
        os.rmdir(temp_dir)
        
        return results
    
    except Exception as e:
        logger.error(f"Error downloading or analyzing file: {e}", exc_info=True)
        return {
            'error': str(e),
            'file_path': f"s3://{bucket}/{key}"
        }


def print_analysis_summary(results):
    """Print a summary of the analysis results.
    
    Args:
        results: Dictionary with analysis results
    """
    print("\n" + "="*80)
    print(f"FNIRS DATA ANALYSIS SUMMARY")
    print("="*80)
    
    if 'error' in results:
        print(f"Error analyzing {results['file_path']}: {results['error']}")
        return
    
    print(f"File: {os.path.basename(results['file_path'])}")
    
    if results['dataset_path']:
        print(f"Dataset: {results['dataset_path']}")
        print(f"Shape: {results['dataset_shape']}")
        print(f"Size: {results['dataset_size_gb']:.2f} GB")
        print(f"Total elements: {results['total_elements']:,}")
        
        print("\nValue Distribution:")
        print(f"  NaN values: {results['nan_count']:,} ({results['nan_percentage']:.2f}%)")
        print(f"  Zero values: {results['zero_count']:,} ({results['zero_percentage']:.2f}%)")
        print(f"  Non-zero values: {results['non_zero_count']:,} ({results['non_zero_percentage']:.2f}%)")
        
        if results['stats']:
            print("\nStatistics (on sample of non-NaN values):")
            print(f"  Min value: {results['stats']['min']}")
            print(f"  Max value: {results['stats']['max']}")
            print(f"  Mean value: {results['stats']['mean']}")
            print(f"  Median value: {results['stats']['median']}")
            print(f"  Standard deviation: {results['stats']['std']}")
    else:
        print("No fNIRS dataset found in the file")
    
    print("="*80)


def main():
    """Main function to run the script."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Analyze fNIRS data in H5 files")
    parser.add_argument("--file", help="Path to local H5 file")
    parser.add_argument("--s3", action="store_true", help="Load from S3 instead of local file")
    parser.add_argument("--bucket", default="conduit-data-dev", help="S3 bucket name")
    parser.add_argument("--key", help="S3 object key")
    parser.add_argument("--chunk-size", type=int, default=100, help="Number of frames to process at a time")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()
    
    # Set log level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Check arguments
    if args.s3 and not args.key:
        parser.error("--s3 requires --key")
    
    if not args.s3 and not args.file:
        parser.error("Either provide --file or use --s3 with --key")
    
    # Analyze the file
    if args.s3:
        results = download_and_analyze_s3_h5(args.bucket, args.key, args.chunk_size)
    else:
        results = analyze_fnirs_in_h5(args.file, args.chunk_size)
    
    # Print the results
    print_analysis_summary(results)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())