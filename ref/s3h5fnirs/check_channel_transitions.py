#!/usr/bin/env python3
"""
Script to check for channels that transition between valid and -inf values across frames.

This script:
1. Downloads an H5 file from S3
2. Analyzes each channel across multiple frames
3. Identifies channels that switch between valid and -inf values
4. Reports statistics and examples of transitioning channels

Usage:
python check_channel_transitions.py --key "curated-h5/filename.h5" --frames 20
"""

import os
import sys
import argparse
import logging
import tempfile
import h5py
import numpy as np
from pathlib import Path
from collections import defaultdict

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("channel_transitions")


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


def check_channel_transitions(file_path, num_frames=20):
    """Check for channels that transition between valid and -inf values.
    
    Args:
        file_path: Path to the H5 file
        num_frames: Number of frames to analyze
        
    Returns:
        Dictionary with analysis results
    """
    logger.info(f"Checking channel transitions in {file_path}")
    
    # Generate channel info
    logger.info("Generating channel information...")
    channel_info = generate_channel_info()
    
    results = {
        'file_path': file_path,
        'file_name': os.path.basename(file_path),
        'frames_analyzed': 0,
        'total_channels': len(channel_info),
        'always_valid_channels': 0,
        'always_inf_channels': 0,
        'transitioning_channels': 0,
        'transition_examples': [],
        'sd_pair_transitions': {}
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
            
            # Get total frames in dataset
            total_frames = dataset.shape[0]
            results['total_frames'] = total_frames
            
            # Determine frames to analyze (evenly spaced)
            frames_to_analyze = min(num_frames, total_frames)
            frame_indices = np.linspace(0, total_frames-1, frames_to_analyze, dtype=int)
            results['frames_analyzed'] = len(frame_indices)
            logger.info(f"Analyzing {frames_to_analyze} frames out of {total_frames} total")
            
            # Initialize arrays to track channel status across frames
            channel_valid = np.zeros((248832, frames_to_analyze), dtype=bool)
            channel_inf = np.zeros((248832, frames_to_analyze), dtype=bool)
            
            # Track source-detector pair transitions
            sd_transitions = defaultdict(lambda: {'valid_frames': [], 'inf_frames': []})
            
            # Analyze each frame
            for i, frame_idx in enumerate(frame_indices):
                logger.info(f"Processing frame {frame_idx+1}/{total_frames} (index {i+1}/{frames_to_analyze})")
                
                # Read frame
                frame = dataset[frame_idx]
                
                # Reshape if needed
                if len(frame.shape) > 1:
                    frame = frame.reshape(-1)
                
                # Mark valid and -inf values
                is_valid = ~np.isneginf(frame)
                is_inf = np.isneginf(frame)
                
                # Update tracking arrays
                channel_valid[:, i] = is_valid
                channel_inf[:, i] = is_inf
                
                # Track source-detector pair status
                for channel_idx in range(248832):
                    if is_valid[channel_idx]:
                        sd_key = channel_info[channel_idx]['sd_key']
                        if frame_idx not in sd_transitions[sd_key]['valid_frames']:
                            sd_transitions[sd_key]['valid_frames'].append(frame_idx)
                    elif is_inf[channel_idx]:
                        sd_key = channel_info[channel_idx]['sd_key']
                        if frame_idx not in sd_transitions[sd_key]['inf_frames']:
                            sd_transitions[sd_key]['inf_frames'].append(frame_idx)
            
            # Identify transitioning channels
            always_valid = np.all(channel_valid, axis=1)
            always_inf = np.all(channel_inf, axis=1)
            transitioning = ~(always_valid | always_inf)
            
            # Count channel categories
            results['always_valid_channels'] = int(np.sum(always_valid))
            results['always_inf_channels'] = int(np.sum(always_inf))
            results['transitioning_channels'] = int(np.sum(transitioning))
            
            # Find examples of transitioning channels
            if results['transitioning_channels'] > 0:
                transition_indices = np.where(transitioning)[0]
                
                # Collect up to 10 examples
                for idx in transition_indices[:min(10, len(transition_indices))]:
                    # Get channel info
                    ch_info = channel_info[idx]
                    
                    # Get frames where it's valid vs. -inf
                    valid_frames = [int(frame_indices[j]) for j in range(frames_to_analyze) if channel_valid[idx, j]]
                    inf_frames = [int(frame_indices[j]) for j in range(frames_to_analyze) if channel_inf[idx, j]]
                    
                    # Create example
                    example = {
                        'channel_idx': int(idx),
                        'channel_name': ch_info['name'],
                        'wavelength': ch_info['wavelength_name'],
                        'moment': ch_info['moment_name'],
                        'source': f"M{ch_info['source_module']}-ID{ch_info['source_id']}",
                        'detector': f"M{ch_info['detector_module']}-ID{ch_info['detector_id']}",
                        'valid_frames': valid_frames,
                        'inf_frames': inf_frames,
                        'sd_key': ch_info['sd_key']
                    }
                    results['transition_examples'].append(example)
            
            # Find source-detector pairs with transitions
            transitioning_sd_pairs = []
            for sd_key, frames in sd_transitions.items():
                if frames['valid_frames'] and frames['inf_frames']:
                    transitioning_sd_pairs.append({
                        'source_module': sd_key[0],
                        'source_id': sd_key[1],
                        'detector_module': sd_key[2],
                        'detector_id': sd_key[3],
                        'valid_frames': frames['valid_frames'],
                        'inf_frames': frames['inf_frames']
                    })
            
            results['transitioning_sd_pairs'] = transitioning_sd_pairs
            results['transitioning_sd_pair_count'] = len(transitioning_sd_pairs)
    
    except Exception as e:
        logger.error(f"Error checking channel transitions: {e}", exc_info=True)
    
    return results


def download_and_analyze_h5(bucket, key, num_frames=20):
    """Download an H5 file from S3 and check for channel transitions.
    
    Args:
        bucket: S3 bucket name
        key: S3 object key
        num_frames: Number of frames to analyze
        
    Returns:
        Dictionary with analysis results
    """
    # Add parent directory to path to import utils
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
    from utils.aws import init_s3_client
    
    logger.info(f"Downloading and analyzing s3://{bucket}/{key}")
    
    # Create temporary directory for download
    temp_dir = tempfile.mkdtemp(prefix="channel_transitions_")
    local_path = os.path.join(temp_dir, os.path.basename(key))
    
    try:
        # Initialize S3 client
        s3_client = init_s3_client()
        
        # Download the file
        logger.info(f"Downloading to {local_path}")
        s3_client.download_file(bucket, key, local_path)
        logger.info(f"Downloaded {os.path.getsize(local_path)/1024/1024:.2f} MB")
        
        # Check for channel transitions
        results = check_channel_transitions(local_path, num_frames)
        
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
    print(f"FNIRS CHANNEL TRANSITION ANALYSIS SUMMARY")
    print("="*80)
    
    if 'error' in results:
        print(f"Error analyzing {results['file_path']}: {results['error']}")
        return
    
    print(f"File: {results['file_name']}")
    print(f"Total Frames in Dataset: {results['total_frames']}")
    print(f"Frames Analyzed: {results['frames_analyzed']}")
    
    # Print channel statistics
    print("\nChannel Status Across Frames:")
    print(f"  Total Channels: {results['total_channels']}")
    print(f"  Always Valid Channels: {results['always_valid_channels']} ({results['always_valid_channels']/results['total_channels']*100:.2f}%)")
    print(f"  Always -Inf Channels: {results['always_inf_channels']} ({results['always_inf_channels']/results['total_channels']*100:.2f}%)")
    print(f"  Transitioning Channels: {results['transitioning_channels']} ({results['transitioning_channels']/results['total_channels']*100:.2f}%)")
    
    # Print transitioning source-detector pair information
    if 'transitioning_sd_pairs' in results:
        print(f"\nSource-Detector Pairs with Transitions: {results['transitioning_sd_pair_count']}")
        
        # Print examples of transitioning source-detector pairs
        if results['transitioning_sd_pairs']:
            print("\nExamples of Transitioning Source-Detector Pairs:")
            for i, pair in enumerate(results['transitioning_sd_pairs'][:5]):
                print(f"  Pair {i+1}: Source (M{pair['source_module']},ID{pair['source_id']}), "
                      f"Detector (M{pair['detector_module']},ID{pair['detector_id']})")
                print(f"    Valid in {len(pair['valid_frames'])} frames: {pair['valid_frames'][:5]}{' ...' if len(pair['valid_frames']) > 5 else ''}")
                print(f"    -Inf in {len(pair['inf_frames'])} frames: {pair['inf_frames'][:5]}{' ...' if len(pair['inf_frames']) > 5 else ''}")
            
            if len(results['transitioning_sd_pairs']) > 5:
                print(f"    ... and {len(results['transitioning_sd_pairs'])-5} more transitioning pairs")
    
    # Print example transitioning channels
    if results['transition_examples']:
        print("\nExamples of Transitioning Channels:")
        for i, example in enumerate(results['transition_examples']):
            print(f"  Channel {i+1}: {example['channel_name']}")
            print(f"    Source: {example['source']}, Detector: {example['detector']}")
            print(f"    Valid in {len(example['valid_frames'])} frames: {example['valid_frames']}")
            print(f"    -Inf in {len(example['inf_frames'])} frames: {example['inf_frames']}")
    else:
        print("\nNo transitioning channels found. All channels are either always valid or always -inf.")
    
    print("="*80)


def main():
    """Main function to run the script."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Check for channels that transition between valid and -inf values")
    parser.add_argument("--bucket", default="conduit-data-dev", help="S3 bucket name")
    parser.add_argument("--key", required=True, help="S3 object key")
    parser.add_argument("--frames", type=int, default=20, help="Number of frames to analyze")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()
    
    # Set log level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Check for channel transitions
    results = download_and_analyze_h5(args.bucket, args.key, args.frames)
    
    # Print summary
    print_analysis_summary(results)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())