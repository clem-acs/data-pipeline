#!/usr/bin/env python3
"""
Script to check if channels containing -inf values are entirely -inf.

This script:
1. Samples frames from the fNIRS dataset in an H5 file
2. For each frame, identifies channels that have at least one -inf value
3. Checks if those channels are entirely composed of -inf values
4. Reports statistics on channel consistency

Usage:
python check_channel_infinity.py --s3 --key "curated-h5/filename.h5" --sample-frames 20
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
logger = logging.getLogger("channel_infinity_check")


def check_channel_infinity(file_path, sample_frames=20):
    """Check if channels with -inf values are entirely -inf.
    
    Args:
        file_path: Path to the H5 file
        sample_frames: Number of frames to sample
        
    Returns:
        Dictionary with analysis results
    """
    logger.info(f"Checking channel infinity in {file_path}")
    
    results = {
        'file_path': file_path,
        'dataset_path': None,
        'dataset_shape': None,
        'frames_analyzed': 0,
        'channels_analyzed': 0,
        'channels_with_inf': 0,
        'fully_inf_channels': 0,
        'mixed_value_channels': 0,
        'channel_details': []
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
                
                # Get number of channels (assuming last dimension is 1)
                num_channels = dataset.shape[1]
                results['channels_analyzed'] = num_channels
                
                # Track channels with -inf values
                channels_with_inf = 0
                fully_inf_channels = 0
                mixed_value_channels = 0
                
                # Channel details for mixed-value channels
                channel_details = []
                
                # Analyze each sampled frame
                for frame_idx in frame_indices:
                    logger.info(f"Processing frame {frame_idx + 1}/{len(frame_indices)}")
                    
                    # Read frame
                    frame = dataset[frame_idx]
                    
                    # Reshape to handle the last dimension (assuming it's 1)
                    if frame.shape[-1] == 1:
                        frame = frame.reshape(frame.shape[0])
                    
                    # Check each channel in the frame
                    for channel_idx in range(num_channels):
                        # Get channel value
                        channel_value = frame[channel_idx]
                        
                        # Check if channel has -inf
                        if np.isneginf(channel_value):
                            channels_with_inf += 1
                            fully_inf_channels += 1
                        elif isinstance(channel_value, np.ndarray):
                            # If channel value is an array, check each element
                            channel_has_inf = np.any(np.isneginf(channel_value))
                            if channel_has_inf:
                                channels_with_inf += 1
                                
                                # Check if the entire channel is -inf
                                all_inf = np.all(np.isneginf(channel_value))
                                if all_inf:
                                    fully_inf_channels += 1
                                else:
                                    mixed_value_channels += 1
                                    
                                    # Collect details for channels with mixed values
                                    inf_count = np.count_nonzero(np.isneginf(channel_value))
                                    total_elements = channel_value.size
                                    inf_percentage = (inf_count / total_elements) * 100
                                    
                                    channel_details.append({
                                        'frame_idx': frame_idx,
                                        'channel_idx': channel_idx,
                                        'total_elements': total_elements,
                                        'inf_count': inf_count,
                                        'inf_percentage': inf_percentage
                                    })
                
                # Update results
                results['frames_analyzed'] = len(frame_indices)
                results['channels_with_inf'] = channels_with_inf
                results['fully_inf_channels'] = fully_inf_channels
                results['mixed_value_channels'] = mixed_value_channels
                results['channel_details'] = channel_details
            else:
                logger.warning("No 'devices/fnirs/frames_data' found in the H5 file")
    
    except Exception as e:
        logger.error(f"Error analyzing H5 file: {e}", exc_info=True)
    
    return results


def download_and_analyze_s3_h5(bucket, key, sample_frames=20):
    """Download an H5 file from S3 and check channel infinity.
    
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
    temp_dir = tempfile.mkdtemp(prefix="channel_infinity_check_")
    local_path = os.path.join(temp_dir, os.path.basename(key))
    
    try:
        # Initialize S3 client
        s3_client = init_s3_client()
        
        # Download the file
        logger.info(f"Downloading to {local_path}")
        s3_client.download_file(bucket, key, local_path)
        logger.info(f"Downloaded {os.path.getsize(local_path)/1024/1024:.2f} MB")
        
        # Analyze the file
        results = check_channel_infinity(local_path, sample_frames)
        
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
    print(f"CHANNEL INFINITY CHECK SUMMARY")
    print("="*80)
    
    if 'error' in results:
        print(f"Error analyzing {results['file_path']}: {results['error']}")
        return
    
    print(f"File: {os.path.basename(results['file_path'])}")
    
    if results['dataset_path']:
        print(f"Dataset: {results['dataset_path']}")
        print(f"Shape: {results['dataset_shape']}")
        print(f"Frames Analyzed: {results['frames_analyzed']}")
        print(f"Channels per Frame: {results['channels_analyzed']}")
        
        # Calculate total channels analyzed
        total_channels = results['frames_analyzed'] * results['channels_analyzed']
        print(f"Total Channels Analyzed: {total_channels:,}")
        
        # Calculate percentages
        inf_percentage = (results['channels_with_inf'] / total_channels) * 100 if total_channels > 0 else 0
        fully_inf_percentage = (results['fully_inf_channels'] / results['channels_with_inf']) * 100 if results['channels_with_inf'] > 0 else 0
        mixed_percentage = (results['mixed_value_channels'] / results['channels_with_inf']) * 100 if results['channels_with_inf'] > 0 else 0
        
        print("\nChannel Analysis:")
        print(f"  Channels with any -inf values: {results['channels_with_inf']:,} ({inf_percentage:.2f}% of all channels)")
        print(f"  Channels that are entirely -inf: {results['fully_inf_channels']:,} ({fully_inf_percentage:.2f}% of channels with -inf)")
        print(f"  Channels with mixed values (some -inf, some finite): {results['mixed_value_channels']:,} ({mixed_percentage:.2f}% of channels with -inf)")
        
        # Show details of mixed-value channels
        if results['mixed_value_channels'] > 0:
            print("\nDetails of Channels with Mixed Values:")
            for i, channel in enumerate(results['channel_details'][:10]):  # Show first 10 only
                print(f"  Channel {i+1}: Frame {channel['frame_idx']}, Channel {channel['channel_idx']}")
                print(f"    -inf values: {channel['inf_count']:,} of {channel['total_elements']:,} ({channel['inf_percentage']:.2f}%)")
            
            if len(results['channel_details']) > 10:
                print(f"  ... and {len(results['channel_details']) - 10} more channels with mixed values")
        
        # Conclusion
        if fully_inf_percentage == 100:
            print("\nCONCLUSION: All channels containing -inf values are entirely composed of -inf values.")
        else:
            print("\nCONCLUSION: Some channels contain a mix of -inf and finite values.")
    else:
        print("No fNIRS dataset found in the file")
    
    print("="*80)


def main():
    """Main function to run the script."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Check if channels with -inf values are entirely -inf")
    parser.add_argument("--file", help="Path to local H5 file")
    parser.add_argument("--s3", action="store_true", help="Load from S3 instead of local file")
    parser.add_argument("--bucket", default="conduit-data-dev", help="S3 bucket name")
    parser.add_argument("--key", help="S3 object key")
    parser.add_argument("--sample-frames", type=int, default=20, help="Number of frames to sample")
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
        results = check_channel_infinity(args.file, args.sample_frames)
    
    # Print the results
    print_analysis_summary(results)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())