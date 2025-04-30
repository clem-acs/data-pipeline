#!/usr/bin/env python3
"""
Script to analyze patterns in -inf values in fNIRS channels.

This script:
1. Downloads an H5 file from S3
2. Analyzes the pattern of valid vs -inf channels
3. Tests the assumption that -inf values are consistent across wavelengths and moments
   for the same source-detector pairs
4. Generates detailed statistics about channel grouping patterns

Usage:
python check_channel_patterns.py --key "curated-h5/filename.h5" --sample-frames 5
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
from collections import defaultdict

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("channel_patterns")


def generate_channel_map():
    """Generate a map of channel indices to source-detector info.
    
    Returns:
        Tuple containing:
        - List of channel names
        - Dictionary mapping channel indices to (source, detector) tuples
        - Dictionary mapping (source, detector) tuples to lists of channel indices
    """
    # Generate all 248,832 channel names and mappings
    channels = []
    channel_to_sd = {}  # Maps channel index to (source module/id, detector module/id)
    sd_to_channels = defaultdict(list)  # Maps (source, detector) to list of channel indices
    
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
                            channels.append(channel_name)
                            
                            # Create source and detector identifiers
                            source = (s_module, s_id)
                            detector = (d_module, d_id)
                            
                            # Map channel index to source-detector pair
                            channel_to_sd[idx] = (source, detector)
                            
                            # Map source-detector pair to channel index
                            sd_key = (s_module, s_id, d_module, d_id)
                            sd_to_channels[sd_key].append(idx)
                            
                            idx += 1
    
    return channels, channel_to_sd, sd_to_channels


def analyze_channel_patterns(file_path, sample_frames=5):
    """Analyze patterns in -inf values in fNIRS channels.
    
    Args:
        file_path: Path to the H5 file
        sample_frames: Number of frames to sample
        
    Returns:
        Dictionary with analysis results
    """
    logger.info(f"Analyzing channel patterns in {file_path}")
    
    # Generate channel mappings
    logger.info("Generating channel mappings...")
    channels, channel_to_sd, sd_to_channels = generate_channel_map()
    
    results = {
        'file_path': file_path,
        'file_name': os.path.basename(file_path),
        'frames_analyzed': 0,
        'total_channels': len(channels),
        'valid_channels': 0,
        'inf_channels': 0,
        'unique_sd_pairs': len(sd_to_channels),
        'valid_sd_pairs': 0,
        'partially_valid_sd_pairs': 0,
        'fully_inf_sd_pairs': 0,
        'assumption_holds': True,  # Will be set to False if any counter-example is found
        'counter_examples': []
    }
    
    try:
        with h5py.File(file_path, 'r') as f:
            # Check if the expected path exists
            if 'devices/fnirs/frames_data' not in f:
                logger.error("No fNIRS data found at 'devices/fnirs/frames_data'")
                return results
            
            dataset = f['devices/fnirs/frames_data']
            
            # Verify shape matches expectations
            if len(dataset.shape) != 3 or dataset.shape[1] != 248832:
                logger.error(f"Unexpected dataset shape: {dataset.shape}, expected (frames, 248832, 1)")
                return results
            
            # Get total frames
            total_frames = dataset.shape[0]
            
            # Sample frames
            if sample_frames >= total_frames:
                frame_indices = list(range(total_frames))
            else:
                # Use evenly spaced frames for better representation
                frame_indices = np.linspace(0, total_frames-1, sample_frames, dtype=int).tolist()
            
            # Initialize counters
            valid_channel_count = np.zeros(248832, dtype=bool)
            inf_channel_count = np.zeros(248832, dtype=bool)
            
            # Track stats for each source-detector pair
            sd_pair_stats = {}
            for sd_key in sd_to_channels:
                sd_pair_stats[sd_key] = {
                    'valid_count': 0,
                    'inf_count': 0,
                    'total_channels': len(sd_to_channels[sd_key]),
                    'channel_indices': sd_to_channels[sd_key]
                }
            
            # Analyze each sampled frame
            for frame_idx in frame_indices:
                logger.info(f"Processing frame {frame_idx+1}/{len(frame_indices)}")
                
                # Read frame
                frame = dataset[frame_idx]
                
                # Reshape to 1D if needed
                if len(frame.shape) > 1:
                    frame = frame.reshape(-1)
                
                # Mark valid and -inf channels
                is_valid = ~np.isneginf(frame)
                is_inf = np.isneginf(frame)
                
                valid_channel_count = valid_channel_count | is_valid
                inf_channel_count = inf_channel_count | is_inf
                
                # Check patterns within source-detector pairs
                for sd_key, channel_indices in sd_to_channels.items():
                    # Get valid/inf status for all channels in this pair
                    sd_valid = is_valid[channel_indices]
                    sd_inf = is_inf[channel_indices]
                    
                    # Update counters
                    if np.any(sd_valid):
                        sd_pair_stats[sd_key]['valid_count'] += 1
                    
                    if np.any(sd_inf):
                        sd_pair_stats[sd_key]['inf_count'] += 1
                    
                    # Check if all channels in this pair have the same status (all valid or all -inf)
                    if np.any(sd_valid) and np.any(sd_inf):
                        # This source-detector pair has mixed statuses
                        results['assumption_holds'] = False
                        
                        # Record counter-example
                        counter_example = {
                            'frame': frame_idx,
                            'sd_key': sd_key,
                            'valid_channels': np.where(sd_valid)[0].tolist(),
                            'inf_channels': np.where(sd_inf)[0].tolist()
                        }
                        results['counter_examples'].append(counter_example)
                        
                        # Limit number of counter-examples
                        if len(results['counter_examples']) >= 10:
                            break
                
                # Stop after finding counter-examples
                if len(results['counter_examples']) >= 10:
                    break
            
            # Calculate overall channel statistics
            results['frames_analyzed'] = len(frame_indices)
            results['valid_channels'] = int(np.sum(valid_channel_count))
            results['inf_channels'] = int(np.sum(inf_channel_count))
            
            # Calculate source-detector pair statistics
            for sd_key, stats in sd_pair_stats.items():
                # A pair is valid if all its channels were valid in at least one frame
                if stats['valid_count'] > 0 and all(valid_channel_count[stats['channel_indices']]):
                    results['valid_sd_pairs'] += 1
                # A pair is partially valid if some but not all channels were valid
                elif stats['valid_count'] > 0:
                    results['partially_valid_sd_pairs'] += 1
                # A pair is fully -inf if all its channels were -inf in all frames
                elif stats['inf_count'] == len(frame_indices):
                    results['fully_inf_sd_pairs'] += 1
            
            # Create detailed results for valid source-detector pairs
            valid_pairs = []
            for sd_key, stats in sd_pair_stats.items():
                if stats['valid_count'] > 0:
                    valid_pairs.append({
                        'source_module': sd_key[0],
                        'source_id': sd_key[1],
                        'detector_module': sd_key[2],
                        'detector_id': sd_key[3],
                        'channels': stats['channel_indices'],
                        'all_valid': all(valid_channel_count[stats['channel_indices']])
                    })
            
            results['valid_pair_details'] = valid_pairs
    
    except Exception as e:
        logger.error(f"Error analyzing channel patterns: {e}", exc_info=True)
    
    return results


def download_and_analyze_h5(bucket, key, sample_frames=5):
    """Download an H5 file from S3 and analyze channel patterns.
    
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
    temp_dir = tempfile.mkdtemp(prefix="channel_patterns_")
    local_path = os.path.join(temp_dir, os.path.basename(key))
    
    try:
        # Initialize S3 client
        s3_client = init_s3_client()
        
        # Download the file
        logger.info(f"Downloading to {local_path}")
        s3_client.download_file(bucket, key, local_path)
        logger.info(f"Downloaded {os.path.getsize(local_path)/1024/1024:.2f} MB")
        
        # Analyze channel patterns
        results = analyze_channel_patterns(local_path, sample_frames)
        
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
    print(f"FNIRS CHANNEL PATTERN ANALYSIS SUMMARY")
    print("="*80)
    
    if 'error' in results:
        print(f"Error analyzing {results['file_path']}: {results['error']}")
        return
    
    print(f"File: {results['file_name']}")
    print(f"Frames Analyzed: {results['frames_analyzed']}")
    
    print("\nChannel Statistics:")
    print(f"  Total Channels: {results['total_channels']}")
    print(f"  Valid Channels: {results['valid_channels']} ({results['valid_channels']/results['total_channels']*100:.2f}%)")
    print(f"  -Inf Channels: {results['inf_channels']} ({results['inf_channels']/results['total_channels']*100:.2f}%)")
    
    print("\nSource-Detector Pair Statistics:")
    print(f"  Total Unique Pairs: {results['unique_sd_pairs']}")
    print(f"  Valid Pairs (all channels valid): {results['valid_sd_pairs']} ({results['valid_sd_pairs']/results['unique_sd_pairs']*100:.2f}%)")
    print(f"  Partially Valid Pairs (some channels valid): {results['partially_valid_sd_pairs']} ({results['partially_valid_sd_pairs']/results['unique_sd_pairs']*100:.2f}%)")
    print(f"  Fully -Inf Pairs (all channels -inf): {results['fully_inf_sd_pairs']} ({results['fully_inf_sd_pairs']/results['unique_sd_pairs']*100:.2f}%)")
    
    print("\nAssumption Testing:")
    if results['assumption_holds']:
        print("  ASSUMPTION HOLDS: For all source-detector pairs, either all channels are valid or all are -inf.")
        print("  This is true across wavelengths (Red/IR) and moments (Zeroth/First/Second).")
    else:
        print("  ASSUMPTION DOES NOT HOLD: Found source-detector pairs with mixed valid and -inf channels.")
        print("\n  Counter-examples found:")
        for i, example in enumerate(results['counter_examples'][:5]):
            sd_key = example['sd_key']
            print(f"    Example {i+1}: Frame {example['frame']}, Source (M{sd_key[0]},ID{sd_key[1]}), Detector (M{sd_key[2]},ID{sd_key[3]})")
            print(f"      Valid channels: {len(example['valid_channels'])}, -Inf channels: {len(example['inf_channels'])}")
        
        if len(results['counter_examples']) > 5:
            print(f"    ... and {len(results['counter_examples'])-5} more counter-examples.")
    
    if 'valid_pair_details' in results:
        print("\nValid Source-Detector Pairs:")
        for i, pair in enumerate(results['valid_pair_details'][:10]):
            print(f"  Pair {i+1}: Source (M{pair['source_module']},ID{pair['source_id']}), Detector (M{pair['detector_module']},ID{pair['detector_id']})")
            print(f"    All channels valid: {'Yes' if pair['all_valid'] else 'No'}")
        
        if len(results['valid_pair_details']) > 10:
            print(f"  ... and {len(results['valid_pair_details'])-10} more valid pairs.")
    
    print("="*80)


def main():
    """Main function to run the script."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Analyze patterns in -inf values in fNIRS channels")
    parser.add_argument("--bucket", default="conduit-data-dev", help="S3 bucket name")
    parser.add_argument("--key", required=True, help="S3 object key")
    parser.add_argument("--sample-frames", type=int, default=5, help="Number of frames to sample")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()
    
    # Set log level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Analyze channel patterns
    results = download_and_analyze_h5(args.bucket, args.key, args.sample_frames)
    
    # Print summary
    print_analysis_summary(results)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())