#!/usr/bin/env python3
"""
Script to specifically check for infinity values in an fNIRS H5 file.
"""
import os
import sys
import h5py
import numpy as np
import tempfile
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("check_infinity")

def check_infinity_in_h5(file_path, chunk_size=100):
    """Check specifically for infinity values in an H5 file.
    
    Args:
        file_path: Path to the H5 file
        chunk_size: Number of frames to process at a time
        
    Returns:
        Dictionary with analysis results
    """
    logger.info(f"Analyzing fNIRS data in {file_path} for infinity values")
    
    results = {
        'file_path': file_path,
        'file_name': os.path.basename(file_path),
        'dataset_path': None,
        'dataset_shape': None,
        'total_elements': 0,
        'inf_count': 0,
        'neg_inf_count': 0,
        'inf_percentage': 0,
        'first_inf_locations': []
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
                
                # Process the dataset in chunks along the first dimension
                total_elements = 0
                inf_count = 0
                neg_inf_count = 0
                max_inf_locations = 20  # Maximum number of infinity locations to record
                
                # Process data in chunks
                for i in range(0, dataset.shape[0], chunk_size):
                    end_idx = min(i + chunk_size, dataset.shape[0])
                    logger.info(f"Processing frames {i} to {end_idx} of {dataset.shape[0]}")
                    
                    # Read chunk
                    chunk = dataset[i:end_idx]
                    
                    # Count infinity values
                    is_inf = np.isinf(chunk)
                    is_neg_inf = np.isneginf(chunk)
                    
                    chunk_inf_count = np.sum(is_inf)
                    chunk_neg_inf_count = np.sum(is_neg_inf)
                    
                    # Update totals
                    total_elements += chunk.size
                    inf_count += chunk_inf_count
                    neg_inf_count += chunk_neg_inf_count
                    
                    # If we haven't recorded enough infinity locations yet, find some in this chunk
                    if len(results['first_inf_locations']) < max_inf_locations and chunk_inf_count > 0:
                        # Find indices of infinity values in this chunk
                        inf_indices = np.where(is_inf)
                        
                        # Convert to absolute indices in the dataset
                        for j in range(min(len(inf_indices[0]), max_inf_locations - len(results['first_inf_locations']))):
                            frame_idx = i + inf_indices[0][j]
                            channel_idx = inf_indices[1][j]
                            other_idx = inf_indices[2][j] if len(inf_indices) > 2 else 0
                            
                            # Record location
                            results['first_inf_locations'].append({
                                'frame': int(frame_idx),
                                'channel': int(channel_idx),
                                'other_idx': int(other_idx)
                            })
                
                # Update results
                results['total_elements'] = total_elements
                results['inf_count'] = int(inf_count)
                results['neg_inf_count'] = int(neg_inf_count)
                
                # Calculate percentage
                if total_elements > 0:
                    results['inf_percentage'] = (inf_count / total_elements) * 100
            else:
                logger.warning("No 'devices/fnirs/frames_data' found in the H5 file")
    
    except Exception as e:
        logger.error(f"Error analyzing H5 file: {e}", exc_info=True)
    
    return results

def download_and_check_s3_h5(bucket, key, chunk_size=100):
    """Download an H5 file from S3 and check for infinity values.
    
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
    temp_dir = tempfile.mkdtemp(prefix="infinity_check_")
    local_path = os.path.join(temp_dir, os.path.basename(key))
    
    try:
        # Initialize S3 client
        s3_client = init_s3_client()
        
        # Download the file
        logger.info(f"Downloading to {local_path}")
        s3_client.download_file(bucket, key, local_path)
        logger.info(f"Downloaded {os.path.getsize(local_path)/1024/1024:.2f} MB")
        
        # Analyze the file
        results = check_infinity_in_h5(local_path, chunk_size)
        
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

def print_infinity_analysis(results):
    """Print a summary of the infinity analysis results.
    
    Args:
        results: Dictionary with analysis results
    """
    print("\n" + "="*80)
    print(f"FNIRS INFINITY ANALYSIS SUMMARY")
    print("="*80)
    
    if 'error' in results:
        print(f"Error analyzing {results['file_path']}: {results['error']}")
        return
    
    print(f"File: {results['file_name']}")
    
    if results['dataset_path']:
        print(f"Dataset: {results['dataset_path']}")
        print(f"Shape: {results['dataset_shape']}")
        print(f"Total elements: {results['total_elements']:,}")
        
        print("\nInfinity Values:")
        print(f"  +Inf values: {results['inf_count']:,} ({results['inf_percentage']:.6f}%)")
        print(f"  -Inf values: {results['neg_inf_count']:,}")
        print(f"  Total Inf values: {results['inf_count'] + results['neg_inf_count']:,}")
        
        if results['first_inf_locations']:
            print("\nSample Infinity Locations:")
            for i, loc in enumerate(results['first_inf_locations'][:10], 1):
                print(f"  {i}. Frame {loc['frame']}, Channel {loc['channel']}, Index {loc['other_idx']}")
    else:
        print("No fNIRS dataset found in the file")
    
    print("="*80)

def main():
    """Main function to run the script."""
    # Parse command-line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description="Check for infinity values in fNIRS H5 files")
    parser.add_argument("--bucket", default="conduit-data-dev", help="S3 bucket name")
    parser.add_argument("--key", help="S3 object key")
    parser.add_argument("--chunk-size", type=int, default=100, help="Number of frames to process at a time")
    parser.add_argument("--file", help="Path to local H5 file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()
    
    # Set log level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Check arguments
    if args.key:
        # Use S3
        results = download_and_check_s3_h5(args.bucket, args.key, args.chunk_size)
    elif args.file:
        # Use local file
        results = check_infinity_in_h5(args.file, args.chunk_size)
    else:
        parser.error("Either --key or --file is required")
        return 1
    
    # Print the results
    print_infinity_analysis(results)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())