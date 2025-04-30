#!/usr/bin/env python3
"""
Script to count channels that have at least one valid (non-inf) value across all frames.

This script:
1. Downloads an H5 file from S3
2. Processes the file in chunks to handle large datasets efficiently
3. Counts channels that have at least one valid value in any frame
4. Reports statistics and details about valid channels

Usage:
python count_ever_valid_channels.py --key "curated-h5/filename.h5" [--chunk-size 100]
"""

import os
import sys
import argparse
import logging
import tempfile
import h5py
import numpy as np
from pathlib import Path
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("ever_valid_channels")


def generate_channel_info():
    """Generate information about channel structure.
    
    Returns:
        Dictionary with channel structure information
    """
    # Create a mapping from channel index to structured channel info
    channel_info = {}
    
    # Loop order must match NumPy's C-order reshaping:
    idx = 0
    for wave_idx in range(2):  # wavelength (slowest)
        wave_name = "Red" if wave_idx == 0 else "IR"
        for moment_idx in range(3):  # moment
            moment_name = ["Zeroth", "First", "Second"][moment_idx]
            for s_module in range(1, 49):  # source module
                for s_id in range(1, 4):  # source id
                    for d_module in range(1, 49):  # detector module
                        for d_id in range(1, 7):  # detector id (fastest)
                            channel_name = f"W{wave_idx}({wave_name})_M{moment_idx}({moment_name})_S{s_module}_{s_id}_D{d_module}_{d_id}"
                            channel_info[idx] = {
                                'name': channel_name,
                                'wavelength_idx': wave_idx,
                                'wavelength_name': wave_name,
                                'moment_idx': moment_idx,
                                'moment_name': moment_name,
                                'source_module': s_module,
                                'source_id': s_id,
                                'detector_module': d_module,
                                'detector_id': d_id,
                                'sd_key': (s_module, s_id, d_module, d_id)
                            }
                            idx += 1
    
    return channel_info


def count_ever_valid_channels(file_path, chunk_size=100, log_interval=500):
    """Count channels that have at least one valid value across all frames.
    
    Args:
        file_path: Path to the H5 file
        chunk_size: Number of frames to process at once
        log_interval: How often to log progress
        
    Returns:
        Dictionary with analysis results
    """
    logger.info(f"Counting ever-valid channels in {file_path}")
    start_time = datetime.now()
    
    # Initialize results
    results = {
        'file_path': file_path,
        'file_name': os.path.basename(file_path),
        'total_frames': 0,
        'total_channels': 248832,
        'ever_valid_channels': 0,
        'always_inf_channels': 0,
        'valid_sd_pairs': set(),
        'channel_info': generate_channel_info(),
        'valid_channel_indices': set()
    }
    
    try:
        with h5py.File(file_path, 'r') as f:
            # Check if the expected path exists
            if 'devices/fnirs/frames_data' not in f:
                logger.error("No fNIRS data found at 'devices/fnirs/frames_data'")
                return results
            
            dataset = f['devices/fnirs/frames_data']
            
            # Verify shape
            if len(dataset.shape) != 3 or dataset.shape[1] != 248832:
                logger.error(f"Unexpected dataset shape: {dataset.shape}, expected (frames, 248832, 1)")
                return results
            
            # Get total frames
            total_frames = dataset.shape[0]
            results['total_frames'] = total_frames
            
            # Initialize mask for channels that have ever been valid
            ever_valid = np.zeros(248832, dtype=bool)
            
            # Process frames in chunks
            for start_idx in range(0, total_frames, chunk_size):
                end_idx = min(start_idx + chunk_size, total_frames)
                
                # Log progress at intervals
                if start_idx % log_interval == 0 or start_idx + chunk_size >= total_frames:
                    elapsed = (datetime.now() - start_time).total_seconds()
                    logger.info(f"Processing frames {start_idx} to {end_idx} of {total_frames} ({start_idx/total_frames*100:.1f}%) - {elapsed:.1f}s elapsed")
                
                # Read chunk of frames
                chunk = dataset[start_idx:end_idx]
                
                # Reshape if needed
                if len(chunk.shape) == 3 and chunk.shape[2] == 1:
                    chunk = chunk.reshape(chunk.shape[0], chunk.shape[1])
                
                # Update ever_valid mask
                # For each channel, check if any frame in this chunk has a valid value
                valid_in_chunk = ~np.isneginf(chunk)
                any_valid_in_chunk = np.any(valid_in_chunk, axis=0)
                ever_valid = ever_valid | any_valid_in_chunk
            
            # Count results
            results['ever_valid_channels'] = int(np.sum(ever_valid))
            results['always_inf_channels'] = 248832 - results['ever_valid_channels']
            
            # Store indices of valid channels
            results['valid_channel_indices'] = set(np.where(ever_valid)[0].tolist())
            
            # Identify valid source-detector pairs
            valid_sd_pairs = set()
            for channel_idx in results['valid_channel_indices']:
                channel = results['channel_info'][channel_idx]
                valid_sd_pairs.add(channel['sd_key'])
            
            results['valid_sd_pairs'] = valid_sd_pairs
            results['valid_sd_pair_count'] = len(valid_sd_pairs)
            
            # Generate summary by source-detector module
            module_summary = {}
            for channel_idx in results['valid_channel_indices']:
                channel = results['channel_info'][channel_idx]
                s_module = channel['source_module']
                d_module = channel['detector_module']
                
                # Use module pair as key
                module_key = (s_module, d_module)
                if module_key not in module_summary:
                    module_summary[module_key] = {
                        'source_module': s_module,
                        'detector_module': d_module,
                        'channel_count': 0,
                        'source_ids': set(),
                        'detector_ids': set(),
                        'sd_pairs': set()
                    }
                
                module_summary[module_key]['channel_count'] += 1
                module_summary[module_key]['source_ids'].add(channel['source_id'])
                module_summary[module_key]['detector_ids'].add(channel['detector_id'])
                module_summary[module_key]['sd_pairs'].add((channel['source_id'], channel['detector_id']))
            
            results['module_summary'] = module_summary
    
    except Exception as e:
        logger.error(f"Error counting ever-valid channels: {e}", exc_info=True)
    
    elapsed_time = (datetime.now() - start_time).total_seconds()
    logger.info(f"Completed analysis in {elapsed_time:.1f} seconds")
    
    return results


def download_and_analyze_h5(bucket, key, chunk_size=100):
    """Download an H5 file from S3 and count ever-valid channels.
    
    Args:
        bucket: S3 bucket name
        key: S3 object key
        chunk_size: Number of frames to process at once
        
    Returns:
        Dictionary with analysis results
    """
    # Add parent directory to path to import utils
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
    from utils.aws import init_s3_client
    
    logger.info(f"Downloading and analyzing s3://{bucket}/{key}")
    start_time = datetime.now()
    
    # Create temporary directory for download
    temp_dir = tempfile.mkdtemp(prefix="ever_valid_channels_")
    local_path = os.path.join(temp_dir, os.path.basename(key))
    
    try:
        # Initialize S3 client
        s3_client = init_s3_client()
        
        # Download the file
        logger.info(f"Downloading to {local_path}")
        s3_client.download_file(bucket, key, local_path)
        logger.info(f"Downloaded {os.path.getsize(local_path)/1024/1024:.2f} MB")
        
        # Count ever-valid channels
        results = count_ever_valid_channels(local_path, chunk_size)
        
        # Clean up temporary file
        os.remove(local_path)
        os.rmdir(temp_dir)
        
        total_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Total processing time: {total_time:.1f} seconds")
        
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
    print(f"FNIRS EVER-VALID CHANNEL ANALYSIS SUMMARY")
    print("="*80)
    
    if 'error' in results:
        print(f"Error analyzing {results['file_path']}: {results['error']}")
        return
    
    print(f"File: {results['file_name']}")
    print(f"Total Frames: {results['total_frames']}")
    
    print("\nChannel Statistics:")
    print(f"  Total Channels: {results['total_channels']}")
    print(f"  Ever Valid Channels: {results['ever_valid_channels']} ({results['ever_valid_channels']/results['total_channels']*100:.2f}%)")
    print(f"  Always -Inf Channels: {results['always_inf_channels']} ({results['always_inf_channels']/results['total_channels']*100:.2f}%)")
    
    print("\nSource-Detector Pair Statistics:")
    print(f"  Total Possible Pairs: 41,472 (48 source modules × 3 sources per module × 48 detector modules × 6 detectors per module)")
    print(f"  Ever Valid Pairs: {results['valid_sd_pair_count']} ({results['valid_sd_pair_count']/41472*100:.2f}%)")
    
    # Print module summary
    if 'module_summary' in results:
        active_modules = len(results['module_summary'])
        print(f"\nActive Module Pairs: {active_modules}")
        
        # Sort by channel count
        sorted_modules = sorted(results['module_summary'].values(), 
                              key=lambda x: x['channel_count'], 
                              reverse=True)
        
        print("\nTop 10 Most Active Module Pairs:")
        for i, module in enumerate(sorted_modules[:10]):
            s_module = module['source_module']
            d_module = module['detector_module']
            channel_count = module['channel_count']
            pair_count = len(module['sd_pairs'])
            source_ids = sorted(list(module['source_ids']))
            detector_ids = sorted(list(module['detector_ids']))
            
            print(f"  {i+1}. Source Module {s_module} to Detector Module {d_module}:")
            print(f"     Channels: {channel_count}")
            print(f"     Source IDs: {source_ids}")
            print(f"     Detector IDs: {detector_ids}")
            print(f"     Active Pairs: {pair_count} of 18 possible (3 sources × 6 detectors)")
        
        if len(sorted_modules) > 10:
            print(f"\n  ... and {len(sorted_modules)-10} more active module pairs")
    
    print("="*80)


def main():
    """Main function to run the script."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Count channels that have at least one valid value across all frames")
    parser.add_argument("--bucket", default="conduit-data-dev", help="S3 bucket name")
    parser.add_argument("--key", required=True, help="S3 object key")
    parser.add_argument("--chunk-size", type=int, default=100, help="Number of frames to process at once")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()
    
    # Set log level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Download and analyze
    results = download_and_analyze_h5(args.bucket, args.key, args.chunk_size)
    
    # Print summary
    print_analysis_summary(results)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())