#!/usr/bin/env python3
"""
Script to calculate source-detector distances for all possible fNIRS channels.

This script takes the layout.json file which contains the 3D positions of sources and detectors,
and calculates the Euclidean distance for each of the 248,832 possible channels.
The distances are output in the same order as the channel indices.
"""

import json
import numpy as np
import argparse
from pathlib import Path


def load_layout_positions(layout_file):
    """
    Load source and detector positions from layout.json.
    
    Args:
        layout_file (str): Path to the layout.json file
        
    Returns:
        tuple: (source_positions, detector_positions)
            - source_positions: numpy array of shape (144, 3) [48 modules x 3 sources per module]
            - detector_positions: numpy array of shape (288, 6) [48 modules x 6 detectors per module]
    """
    with open(layout_file, 'r') as f:
        layout_data = json.load(f)
    
    # Load source positions
    source_locations = layout_data.get("source_locations", [])
    source_positions = np.zeros((48, 3, 3))  # [module, source_id, xyz]
    
    for module_idx, module_sources in enumerate(source_locations):
        if module_idx < 48:  # Ensure we don't exceed array bounds
            for source_id, pos in enumerate(module_sources):
                if source_id < 3:  # Each module has 3 sources
                    source_positions[module_idx, source_id] = pos
    
    # Load detector positions
    detector_locations = layout_data.get("detector_locations", [])
    detector_positions = np.zeros((48, 6, 3))  # [module, detector_id, xyz]
    
    for module_idx, module_detectors in enumerate(detector_locations):
        if module_idx < 48:  # Ensure we don't exceed array bounds
            for detector_id, pos in enumerate(module_detectors[:6]):  # Only consider the first 6 detectors (skip central one)
                detector_positions[module_idx, detector_id] = pos
    
    return source_positions, detector_positions


def calculate_channel_index(wavelength_idx, moment_idx, source_module, source_id, detector_module, detector_id):
    """
    Calculate the index for a specific channel based on the formula:
    
    index = ((((wavelength_idx * 3 + moment_idx) * 48 + (source_module-1)) * 3 + (source_id-1)) * 48 + (detector_module-1)) * 6 + (detector_id-1)
    
    Parameters:
    - wavelength_idx: 0 (Red) or 1 (IR)
    - moment_idx: 0 (Zeroth), 1 (First), or 2 (Second)
    - source_module: 1-48
    - source_id: 1-3
    - detector_module: 1-48
    - detector_id: 1-6
    
    Returns:
    - index: The calculated channel index
    """
    index = ((((wavelength_idx * 3 + moment_idx) * 48 + (source_module-1)) * 3 + (source_id-1)) * 48 + (detector_module-1)) * 6 + (detector_id-1)
    return index


def calculate_all_distances(source_positions, detector_positions, output_csv=None):
    """
    Calculate the Euclidean distance for all 248,832 possible channels.
    
    Args:
        source_positions (numpy.ndarray): Source positions from layout.json
        detector_positions (numpy.ndarray): Detector positions from layout.json
        output_csv (str, optional): Path to output CSV file
        
    Returns:
        numpy.ndarray: Array of shape (248832,) containing the distance for each channel index
    """
    # Initialize array to store distances
    all_distances = np.zeros(248832)
    channel_info = []
    
    # Iterate through all combinations
    for wavelength_idx in range(2):  # 0=Red, 1=IR
        for moment_idx in range(3):  # 0=Zeroth, 1=First, 2=Second
            for source_module in range(1, 49):  # 1-48
                for source_id in range(1, 4):  # 1-3
                    for detector_module in range(1, 49):  # 1-48
                        for detector_id in range(1, 7):  # 1-6
                            # Calculate channel index
                            channel_idx = calculate_channel_index(
                                wavelength_idx, moment_idx,
                                source_module, source_id,
                                detector_module, detector_id
                            )
                            
                            # Calculate Euclidean distance between source and detector
                            source_pos = source_positions[source_module-1, source_id-1]
                            detector_pos = detector_positions[detector_module-1, detector_id-1]
                            
                            distance = np.sqrt(np.sum((source_pos - detector_pos)**2))
                            
                            # Store distance at the channel index
                            all_distances[channel_idx] = distance
                            
                            # For debugging and CSV output
                            channel_name = f"W{wavelength_idx}_M{moment_idx}_S{source_module}_{source_id}_D{detector_module}_{detector_id}"
                            channel_info.append({
                                'index': channel_idx,
                                'name': channel_name,
                                'wavelength': wavelength_idx,
                                'moment': moment_idx,
                                'source_module': source_module,
                                'source_id': source_id,
                                'detector_module': detector_module,
                                'detector_id': detector_id,
                                'distance': distance
                            })
    
    # Sort channel_info by index for CSV output
    channel_info.sort(key=lambda x: x['index'])
    
    # Save to CSV if requested
    if output_csv:
        import csv
        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=channel_info[0].keys())
            writer.writeheader()
            writer.writerows(channel_info)
    
    return all_distances


def generate_distance_filter(all_distances, min_distance=None, max_distance=None):
    """
    Generate a filter for valid channels based on distance thresholds.
    
    Args:
        all_distances (numpy.ndarray): Array of distances for all channels
        min_distance (float, optional): Minimum distance threshold (cm)
        max_distance (float, optional): Maximum distance threshold (cm)
        
    Returns:
        numpy.ndarray: Boolean array indicating valid channels
    """
    valid_channels = np.ones_like(all_distances, dtype=bool)
    
    if min_distance is not None:
        valid_channels = valid_channels & (all_distances >= min_distance)
    
    if max_distance is not None:
        valid_channels = valid_channels & (all_distances <= max_distance)
    
    return valid_channels


def main():
    parser = argparse.ArgumentParser(description='Calculate source-detector distances for fNIRS channels')
    parser.add_argument('--layout', default="/Users/clem/Desktop/code/data-pipeline/ref/layout.json",
                        help='Path to layout.json file')
    parser.add_argument('--output', '-o', help='Output NPY file for distances')
    parser.add_argument('--csv', help='Output CSV file with full channel information')
    parser.add_argument('--min-distance', type=float, help='Minimum valid distance (cm)')
    parser.add_argument('--max-distance', type=float, help='Maximum valid distance (cm)')
    args = parser.parse_args()
    
    # Load positions from layout.json
    source_positions, detector_positions = load_layout_positions(args.layout)
    
    print(f"Loaded positions from {args.layout}")
    print(f"Sources: {source_positions.shape}")
    print(f"Detectors: {detector_positions.shape}")
    
    # Calculate distances for all channels
    print("Calculating distances for all 248,832 channels...")
    all_distances = calculate_all_distances(source_positions, detector_positions, args.csv)
    
    # Generate basic statistics
    print("\nDistance Statistics:")
    print(f"Min distance: {np.min(all_distances):.2f} cm")
    print(f"Max distance: {np.max(all_distances):.2f} cm")
    print(f"Mean distance: {np.mean(all_distances):.2f} cm")
    print(f"Median distance: {np.median(all_distances):.2f} cm")
    
    # Apply distance filter if specified
    if args.min_distance is not None or args.max_distance is not None:
        valid_channels = generate_distance_filter(all_distances, args.min_distance, args.max_distance)
        valid_count = np.sum(valid_channels)
        print(f"\nValid channels (distance filter): {valid_count} ({valid_count/len(all_distances)*100:.2f}%)")
        
        if args.min_distance:
            print(f"  Min distance threshold: {args.min_distance} cm")
        if args.max_distance:
            print(f"  Max distance threshold: {args.max_distance} cm")
    
    # Save distances to NPY file if specified
    if args.output:
        np.save(args.output, all_distances)
        print(f"\nSaved distances to {args.output}")
    
    print("\nDone.")


if __name__ == "__main__":
    main()