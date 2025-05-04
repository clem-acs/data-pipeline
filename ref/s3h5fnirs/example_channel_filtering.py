#!/usr/bin/env python3
"""
Example script showing how to generate and filter fNIRS channel indices
and preprocess fNIRS data.
"""

import json
import numpy as np
import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the project root to the Python path to import from transforms
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from transforms.neural_processing.fnirs_preprocessing import (
    generate_valid_indices, filter_by_distance, preprocess_fnirs, generate_channel_name
)


def main():
    # Example list of modules present in the data
    present_modules = [1, 2, 3, 4, 5]
    print(f"Modules present: {present_modules}")
    
    # Load the layout.json for distances
    layout_path = os.path.join(os.path.dirname(__file__), '../layout.json')
    with open(layout_path, 'r') as f:
        layout_data = json.load(f)
    
    # Generate all valid channel indices for these modules
    all_indices = generate_valid_indices(present_modules)
    print(f"Generated {len(all_indices)} potential channel indices")
    
    # Example 1: Filter by distance (<60mm)
    filtered_indices = filter_by_distance(
        all_indices, layout_data, max_distance=60
    )
    print(f"\nFiltered to distances <60mm: {len(filtered_indices)} channels")
    
    # Example 2: Filter by same module only
    same_module_indices = filter_by_distance(
        all_indices, layout_data, same_module_only=True
    )
    print(f"Filtered to same module only: {len(same_module_indices)} channels")
    
    # Example 3: Combine both filters
    close_same_module = filter_by_distance(
        all_indices, layout_data, max_distance=60, same_module_only=True
    )
    print(f"Channels <60mm in same module: {len(close_same_module)} channels")
    
    # Example 4: Different modules with close distances
    cross_module_close = filter_by_distance(
        all_indices, layout_data, max_distance=60, different_module_only=True
    )
    print(f"Channels <60mm across different modules: {len(cross_module_close)} channels")
    
    # Example 5: Create synthetic fNIRS dataset for testing
    # We'll make a smaller dataset for demonstration purposes
    num_frames = 10
    num_channels = 5000  # Reduced for easier demonstration
    
    # Generate synthetic data where most channels have -inf values
    # but a few have actual values (to test the validation)
    synthetic_data = np.full((num_frames, num_channels, 1), -np.inf)
    
    # Add some valid data for channels that should be kept (feasible channels)
    for idx in filtered_indices[:100]:  # First 100 feasible channels
        if idx < num_channels:
            synthetic_data[:, idx, :] = np.random.randn(num_frames, 1)
    
    # Add some valid data for channels that should be removed (non-feasible)
    # This will trigger the validation warning
    non_feasible_indices = list(set(range(num_channels)) - set(filtered_indices))
    for idx in non_feasible_indices[:10]:  # First 10 non-feasible channels
        synthetic_data[:, idx, :] = np.random.randn(num_frames, 1)
    
    print(f"\nCreated synthetic fNIRS dataset with shape: {synthetic_data.shape}")
    print(f"Added valid data to 100 feasible channels and 10 non-feasible channels")
    
    # Initial metadata
    metadata = {
        'sampling_rate': 10.0,  # Hz
        'session_id': 'example_session',
    }
    
    # Process the data to filter channels by distance
    processed_data, processed_metadata = preprocess_fnirs(
        synthetic_data,
        metadata,
        layout_data=layout_data,
        present_modules=present_modules,
        max_distance_mm=60
    )
    
    print(f"\nAfter preprocessing:")
    print(f"Original data shape: {synthetic_data.shape}")
    print(f"Processed data shape: {processed_data.shape}")
    
    # Check for warnings about non-inf channels
    if 'warning' in processed_metadata:
        print(f"Warning: {processed_metadata['warning']}")
        print(f"Non-inf removed channels: {processed_metadata['non_inf_removed_channels'][:5]}... (and more)")
    
    # Show the metadata
    print("\nMetadata contains:")
    for key in processed_metadata:
        if key in ['feasible_channel_indices', 'feasible_channel_names']:
            print(f"  {key}: {len(processed_metadata[key])} items")
        else:
            print(f"  {key}: {processed_metadata[key]}")
    
    # Display some of the feasible channel names
    print("\nFirst 5 feasible channel names:")
    for i, name in enumerate(processed_metadata['feasible_channel_names'][:5]):
        index = processed_metadata['feasible_channel_indices'][i]
        print(f"  {i+1}. Index {index}: {name}")


if __name__ == "__main__":
    main()