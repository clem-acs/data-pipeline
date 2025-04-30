#!/usr/bin/env python3
"""
Simple script to check for infinite values in fNIRS data in H5 files.

This script analyzes fNIRS data in curated H5 files, 
reporting the percentage of -inf, inf, and finite values.

Usage:
python check_infinity_simple.py --s3 --key "curated-h5/filename.h5" --sample-frames 100
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
logger = logging.getLogger("fnirs_infinity_check")


def check_infinity_in_fnirs(file_path, sample_frames=100):
    """Check for infinite values in fNIRS data.
    
    Args:
        file_path: Path to the H5 file
        sample_frames: Number of frames to sample
        
    Returns:
        Dictionary with analysis results
    """
    logger.info(f"Checking for infinity in fNIRS data in {file_path}")
    
    results = {
        'file_path': file_path,
        'dataset_path': None,
        'dataset_shape': None,
        'total_elements_sampled': 0,
        'neg_inf_count': 0,
        'pos_inf_count': 0,
        'finite_count': 0,
        'neg_inf_percentage': 0,
        'pos_inf_percentage': 0,
        'finite_percentage': 0
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
                
                logger.info(f"Found fNIRS dataset at {dataset_path} with shape {dataset.shape}")
                
                # Sample random frames
                if sample_frames >= dataset.shape[0]:
                    # Analyze all frames
                    frame_indices = np.arange(dataset.shape[0])
                else:
                    # Sample random frames
                    frame_indices = np.random.choice(dataset.shape[0], size=sample_frames, replace=False)
                    frame_indices.sort()  # Sort to ensure sequential reading
                
                # Process sampled frames
                total_elements = 0
                neg_inf_count = 0
                pos_inf_count = 0
                
                for frame_idx in frame_indices:
                    logger.info(f"Processing frame {frame_idx + 1}/{len(frame_indices)}")
                    
                    # Read frame
                    frame = dataset[frame_idx]
                    
                    # Count different types of values
                    neg_inf_mask = np.isneginf(frame)
                    pos_inf_mask = np.isposinf(frame)
                    
                    neg_inf_count += np.count_nonzero(neg_inf_mask)
                    pos_inf_count += np.count_nonzero(pos_inf_mask)
                    
                    total_elements += frame.size
                
                # Calculate finite count
                finite_count = total_elements - neg_inf_count - pos_inf_count
                
                # Update results
                results['total_elements_sampled'] = total_elements
                results['neg_inf_count'] = int(neg_inf_count)
                results['pos_inf_count'] = int(pos_inf_count)
                results['finite_count'] = int(finite_count)
                
                # Calculate percentages
                results['neg_inf_percentage'] = (neg_inf_count / total_elements) * 100
                results['pos_inf_percentage'] = (pos_inf_count / total_elements) * 100
                results['finite_percentage'] = (finite_count / total_elements) * 100
            else:
                logger.warning("No 'devices/fnirs/frames_data' found in the H5 file")
    
    except Exception as e:
        logger.error(f"Error analyzing H5 file: {e}", exc_info=True)
    
    return results


def download_and_analyze_s3_h5(bucket, key, sample_frames=100):
    """Download an H5 file from S3 and check for infinity in its fNIRS data.
    
    Args:
        bucket: S3 bucket name
        key: S3 object key
        sample_frames: Number of frames to sample
        
    Returns:
        Dictionary with analysis results
    """
    # Add parent directory to path to import utils
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
    from utils.aws import init_s3_client
    
    logger.info(f"Downloading and analyzing s3://{bucket}/{key}")
    
    # Create temporary directory for download
    temp_dir = tempfile.mkdtemp(prefix="fnirs_infinity_check_")
    local_path = os.path.join(temp_dir, os.path.basename(key))
    
    try:
        # Initialize S3 client
        s3_client = init_s3_client()
        
        # Download the file
        logger.info(f"Downloading to {local_path}")
        s3_client.download_file(bucket, key, local_path)
        logger.info(f"Downloaded {os.path.getsize(local_path)/1024/1024:.2f} MB")
        
        # Analyze the file
        results = check_infinity_in_fnirs(local_path, sample_frames)
        
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
    print(f"FNIRS INFINITY CHECK SUMMARY")
    print("="*80)
    
    if 'error' in results:
        print(f"Error analyzing {results['file_path']}: {results['error']}")
        return
    
    print(f"File: {os.path.basename(results['file_path'])}")
    
    if results['dataset_path']:
        print(f"Dataset: {results['dataset_path']}")
        print(f"Shape: {results['dataset_shape']}")
        print(f"Total Elements Sampled: {results['total_elements_sampled']:,}")
        
        print("\nValue Distribution:")
        print(f"  Negative infinity (-inf): {results['neg_inf_count']:,} ({results['neg_inf_percentage']:.2f}%)")
        print(f"  Positive infinity (inf): {results['pos_inf_count']:,} ({results['pos_inf_percentage']:.2f}%)")
        print(f"  All infinity: {results['neg_inf_count'] + results['pos_inf_count']:,} ({results['neg_inf_percentage'] + results['pos_inf_percentage']:.2f}%)")
        print(f"  Finite values: {results['finite_count']:,} ({results['finite_percentage']:.2f}%)")
    else:
        print("No fNIRS dataset found in the file")
    
    print("="*80)


def main():
    """Main function to run the script."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Check for infinity in fNIRS data")
    parser.add_argument("--file", help="Path to local H5 file")
    parser.add_argument("--s3", action="store_true", help="Load from S3 instead of local file")
    parser.add_argument("--bucket", default="conduit-data-dev", help="S3 bucket name")
    parser.add_argument("--key", help="S3 object key")
    parser.add_argument("--sample-frames", type=int, default=100, help="Number of frames to sample")
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
        results = download_and_analyze_s3_h5(args.bucket, args.key, args.sample_frames)
    else:
        results = check_infinity_in_fnirs(args.file, args.sample_frames)
    
    # Print the results
    print_analysis_summary(results)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())