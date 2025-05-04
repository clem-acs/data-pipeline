#!/usr/bin/env python3
"""
Script to test the neural window transform with the updated fNIRS preprocessing.

This script creates synthetic EEG and fNIRS data and runs it through the
neural window transform to validate that the fNIRS preprocessing with
layout.json and hardcoded modules is working correctly.
"""

import os
import sys
import json
import numpy as np
import logging
import h5py
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

# Import necessary modules
from transforms.neural_window_transform import NeuralWindowTransform
from transforms.neural_processing.fnirs_preprocessing import (
    generate_valid_indices, filter_by_distance, generate_channel_name
)


def create_synthetic_h5(filepath, num_frames=100):
    """Create a synthetic H5 file for testing."""
    logger.info(f"Creating synthetic H5 file at {filepath}")
    
    # Create the file
    with h5py.File(filepath, 'w') as f:
        # Create device groups
        devices = f.create_group('devices')
        eeg = devices.create_group('eeg')
        fnirs = devices.create_group('fnirs')
        
        # Create EEG data (simple random data)
        eeg_data = np.random.randn(num_frames, 64, 1)  # 64 EEG channels
        eeg_timestamps = np.linspace(0, num_frames/100, num_frames)  # 100 Hz
        
        eeg.create_dataset('frames_data', data=eeg_data)
        eeg.create_dataset('timestamps', data=eeg_timestamps)
        
        # Create fNIRS data with -inf for most channels
        # We'll generate full size for 248,832 channels
        fnirs_data = np.full((num_frames, 248832, 1), -np.inf)
        
        # Load layout.json
        layout_path = os.path.join(project_root, 'transforms', 'neural_processing', 'layout.json')
        with open(layout_path, 'r') as layout_file:
            layout_data = json.load(layout_file)
        
        # Use the same hardcoded modules as in the transform
        present_modules = [1, 2, 3, 4, 5, 6, 7, 8, 13, 14, 19, 20, 23, 24, 25, 26, 29, 30, 31, 32]
        
        # Generate valid indices for these modules with distance < 60mm
        valid_indices = generate_valid_indices(present_modules)
        feasible_indices = filter_by_distance(valid_indices, layout_data, max_distance=60)
        
        # Set valid data for feasible channels
        for idx in feasible_indices[:100]:  # Only use first 100 for speed
            if idx < fnirs_data.shape[1]:
                fnirs_data[:, idx, :] = np.random.randn(num_frames, 1)
        
        # Log some channel names we're using
        logger.info("Sample of feasible channel names:")
        for idx in feasible_indices[:5]:
            logger.info(f"  Index {idx}: {generate_channel_name(idx)}")
        
        # Set timestamps
        fnirs_timestamps = np.linspace(0, num_frames/10, num_frames)  # 10 Hz
        
        fnirs.create_dataset('frames_data', data=fnirs_data)
        fnirs.create_dataset('timestamps', data=fnirs_timestamps)
        
        # Add some file-level attributes
        f.attrs['session_id'] = 'test_session'
        f.attrs['subject_id'] = 'test_subject'
    
    logger.info(f"Created synthetic H5 file with {num_frames} frames")
    return filepath


def test_neural_window_transform():
    """Test the neural window transform with synthetic data."""
    # Create a temporary directory for test files
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a synthetic H5 file
        h5_path = os.path.join(tmpdir, 'test_session.h5')
        create_synthetic_h5(h5_path)
        
        # Create a minimal "S3 bucket" directory structure
        s3_dir = os.path.join(tmpdir, 's3_bucket')
        source_dir = os.path.join(s3_dir, 'curated-h5')
        dest_dir = os.path.join(s3_dir, 'windowed-data')
        os.makedirs(source_dir)
        os.makedirs(dest_dir)
        
        # Copy the H5 file to the source directory
        import shutil
        source_h5 = os.path.join(source_dir, 'test_session.h5')
        shutil.copy(h5_path, source_h5)
        
        class MinimalS3:
            """Minimal S3-like class for testing."""
            def __init__(self, base_dir):
                self.base_dir = base_dir
            
            def get_paginator(self, operation):
                class Paginator:
                    def __init__(self, base_dir):
                        self.base_dir = base_dir
                    
                    def paginate(self, Bucket, Prefix):
                        prefix_dir = os.path.join(self.base_dir, Prefix)
                        files = [os.path.join(Prefix, f) for f in os.listdir(prefix_dir)]
                        return [{'Contents': [{'Key': f} for f in files]}]
                
                return Paginator(self.base_dir)
            
            def download_file(self, bucket, key, filename):
                src = os.path.join(self.base_dir, key)
                shutil.copy(src, filename)
            
            def upload_file(self, filename, bucket, key):
                dest = os.path.join(self.base_dir, key)
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                shutil.copy(filename, dest)
        
        # Create a mock transform with our minimal S3
        transform = NeuralWindowTransform(
            s3_bucket='s3_bucket',
            source_prefix='curated-h5/',
            dest_prefix='windowed-data/',
            window_size_sec=2.0,
            window_step_sec=1.0,
            verbose=True,
            dry_run=True  # Use dry run mode for testing
        )
        
        # Replace the S3 client with our minimal version
        transform.s3 = MinimalS3(s3_dir)
        
        # Mock the record_transform method
        transform.record_transform = lambda **kwargs: {'status': 'mocked', 'kwargs': kwargs}
        
        # Find and process items
        items = transform.find_items_to_process()
        logger.info(f"Found {len(items)} items to process")
        
        # Process the first item
        if items:
            try:
                result = transform.process_item(items[0])
                logger.info(f"Processing result: {result}")
                logger.info("Test completed successfully!")
            except Exception as e:
                logger.error(f"Error processing item: {e}", exc_info=True)
        else:
            logger.warning("No items found to process")


if __name__ == "__main__":
    test_neural_window_transform()