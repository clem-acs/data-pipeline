#!/usr/bin/env python3
"""
Script to analyze fNIRS data in curated H5 files for NaN, zero, and non-zero values.

This script:
1. Loads an H5 file containing fNIRS data
2. Finds the fNIRS dataset in the 'devices/fnirs' group
3. Analyzes what percentage of values are NaN, zero, and non-zero
4. Optionally creates a visualization of the distribution

Usage:
python analyze_fnirs_values.py <path_to_h5_file> [--plot] [--sample-size SIZE]
"""

import os
import sys
import argparse
import logging
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tempfile

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("analyze_fnirs")


def find_fnirs_dataset(h5_file):
    """Find the fNIRS dataset in an H5 file.
    
    Args:
        h5_file: Open h5py File object
        
    Returns:
        Tuple of (dataset, path) or (None, None) if not found
    """
    # Check if the expected path exists
    if 'devices/fnirs/frames_data' in h5_file:
        return h5_file['devices/fnirs/frames_data'], 'devices/fnirs/frames_data'
    
    # If not found at the expected path, search for it
    fnirs_dataset = None
    fnirs_path = None
    
    # Look for any group with 'fnirs' in the name
    for group_name in h5_file:
        if 'fnir' in group_name.lower():
            group = h5_file[group_name]
            
            # If it's a group, look for datasets in it
            if isinstance(group, h5py.Group):
                for subname in group:
                    # Look for datasets with 'data', 'frames', 'timeseries' in the name
                    if any(keyword in subname.lower() for keyword in ['data', 'frames', 'timeseries']):
                        dataset_path = f"{group_name}/{subname}"
                        dataset = h5_file[dataset_path]
                        
                        if isinstance(dataset, h5py.Dataset) and len(dataset.shape) >= 2:
                            logger.info(f"Found fNIRS dataset at {dataset_path} with shape {dataset.shape}")
                            return dataset, dataset_path
    
    # If still not found, check if there's a 'devices' group with a 'fnirs' subgroup
    if 'devices' in h5_file:
        devices = h5_file['devices']
        
        for device_name in devices:
            if 'fnir' in device_name.lower():
                device = devices[device_name]
                
                # Look for dataset in this device group
                for data_name in device:
                    if any(keyword in data_name.lower() for keyword in ['data', 'frames', 'timeseries']):
                        dataset_path = f"devices/{device_name}/{data_name}"
                        dataset = h5_file[dataset_path]
                        
                        if isinstance(dataset, h5py.Dataset) and len(dataset.shape) >= 2:
                            logger.info(f"Found fNIRS dataset at {dataset_path} with shape {dataset.shape}")
                            return dataset, dataset_path
    
    logger.warning("No fNIRS dataset found in the H5 file")
    return None, None


def analyze_fnirs_values(h5_file_path, sample_size=None, create_plot=False):
    """Analyze fNIRS data values in an H5 file.
    
    Args:
        h5_file_path: Path to the H5 file
        sample_size: If provided, analyze only a subset of the data
        create_plot: Whether to create visualization plots
        
    Returns:
        Dictionary with analysis results
    """
    logger.info(f"Analyzing fNIRS data in {h5_file_path}")
    
    results = {
        'file_path': h5_file_path,
        'dataset_path': None,
        'dataset_shape': None,
        'dataset_size_gb': None,
        'analyzed_elements': None,
        'nan_percentage': None,
        'zero_percentage': None,
        'non_zero_percentage': None,
        'min_value': None,
        'max_value': None,
        'mean_value': None,
        'std_value': None,
        'plot_path': None,
        'histogram_path': None
    }
    
    try:
        with h5py.File(h5_file_path, 'r') as f:
            # Find fNIRS dataset
            dataset, dataset_path = find_fnirs_dataset(f)
            
            if dataset is None:
                logger.error("No fNIRS dataset found in the file")
                return results
            
            # Record dataset information
            results['dataset_path'] = dataset_path
            results['dataset_shape'] = dataset.shape
            results['dataset_size_gb'] = dataset.size * dataset.dtype.itemsize / (1024**3)
            
            logger.info(f"Found fNIRS dataset at {dataset_path} with shape {dataset.shape}")
            logger.info(f"Dataset size: {results['dataset_size_gb']:.2f} GB")
            
            # For very large datasets, we may want to analyze only a portion
            if sample_size is not None and sample_size < dataset.size:
                logger.info(f"Analyzing a sample of {sample_size} elements")
                
                # Determine how to sample the data based on its shape
                if len(dataset.shape) == 1:
                    # 1D array - sample random indices
                    indices = np.random.choice(dataset.shape[0], size=sample_size, replace=False)
                    data_sample = dataset[indices]
                elif len(dataset.shape) == 2:
                    # 2D array - sample random rows
                    row_indices = np.random.choice(dataset.shape[0], size=min(sample_size // dataset.shape[1], dataset.shape[0]), replace=False)
                    data_sample = dataset[row_indices, :]
                elif len(dataset.shape) == 3:
                    # 3D array - sample random chunks
                    frames_to_sample = min(1000, dataset.shape[0])
                    channels_to_sample = min(1000, dataset.shape[1])
                    
                    frame_indices = np.random.choice(dataset.shape[0], size=frames_to_sample, replace=False)
                    channel_indices = np.random.choice(dataset.shape[1], size=channels_to_sample, replace=False)
                    
                    # Create a list to collect samples
                    samples = []
                    total_elements = 0
                    
                    for frame_idx in frame_indices:
                        frame_data = dataset[frame_idx, channel_indices, :]
                        samples.append(frame_data.flatten())
                        total_elements += frame_data.size
                        
                        if total_elements >= sample_size:
                            break
                    
                    data_sample = np.concatenate(samples)[:sample_size]
                else:
                    # Higher dimensions - flatten a portion
                    logger.warning(f"Dataset has {len(dataset.shape)} dimensions, using flattened subset")
                    flat_indices = np.random.choice(dataset.size, size=sample_size, replace=False)
                    flat_data = dataset.flatten()[flat_indices]
                    data_sample = flat_data
                
                results['analyzed_elements'] = data_sample.size
                
                # Calculate statistics on the sample
                nan_count = np.isnan(data_sample).sum()
                zero_count = np.logical_and(data_sample == 0, ~np.isnan(data_sample)).sum()
                non_zero_count = data_sample.size - nan_count - zero_count
                
                results['nan_percentage'] = (nan_count / data_sample.size) * 100
                results['zero_percentage'] = (zero_count / data_sample.size) * 100
                results['non_zero_percentage'] = (non_zero_count / data_sample.size) * 100
                
                # Calculate basic statistics on non-NaN values
                valid_data = data_sample[~np.isnan(data_sample)]
                
                if len(valid_data) > 0:
                    results['min_value'] = float(np.min(valid_data))
                    results['max_value'] = float(np.max(valid_data))
                    results['mean_value'] = float(np.mean(valid_data))
                    results['std_value'] = float(np.std(valid_data))
            else:
                # For smaller datasets, analyze the entire dataset
                # Process in chunks to avoid memory issues
                logger.info("Analyzing the entire dataset in chunks")
                
                chunk_size = 1000000  # Adjust based on memory constraints
                total_size = dataset.size
                results['analyzed_elements'] = total_size
                
                nan_count = 0
                zero_count = 0
                
                # For computing statistics on non-NaN values
                all_values = []
                max_sample_size = min(10000000, total_size)  # Cap the sample size for statistics
                sample_ratio = max_sample_size / total_size
                
                # Process 1D, 2D, and 3D arrays differently
                if len(dataset.shape) == 1:
                    # Process 1D arrays in chunks
                    for start_idx in range(0, dataset.shape[0], chunk_size):
                        end_idx = min(start_idx + chunk_size, dataset.shape[0])
                        chunk = dataset[start_idx:end_idx]
                        
                        nan_count += np.isnan(chunk).sum()
                        zero_count += np.logical_and(chunk == 0, ~np.isnan(chunk)).sum()
                        
                        # Sample some values for statistics
                        if np.random.rand() < sample_ratio:
                            all_values.append(chunk[~np.isnan(chunk)])
                        
                elif len(dataset.shape) == 2:
                    # Process 2D arrays in chunks of rows
                    rows_per_chunk = max(1, chunk_size // dataset.shape[1])
                    
                    for start_row in range(0, dataset.shape[0], rows_per_chunk):
                        end_row = min(start_row + rows_per_chunk, dataset.shape[0])
                        chunk = dataset[start_row:end_row, :]
                        
                        nan_count += np.isnan(chunk).sum()
                        zero_count += np.logical_and(chunk == 0, ~np.isnan(chunk)).sum()
                        
                        # Sample some values for statistics
                        if np.random.rand() < sample_ratio:
                            all_values.append(chunk[~np.isnan(chunk)])
                        
                elif len(dataset.shape) == 3:
                    # Process 3D arrays frame by frame
                    for frame in range(dataset.shape[0]):
                        chunk = dataset[frame, :, :]
                        
                        nan_count += np.isnan(chunk).sum()
                        zero_count += np.logical_and(chunk == 0, ~np.isnan(chunk)).sum()
                        
                        # Sample some values for statistics
                        if np.random.rand() < sample_ratio:
                            all_values.append(chunk[~np.isnan(chunk)])
                
                # Calculate final percentages
                results['nan_percentage'] = (nan_count / total_size) * 100
                results['zero_percentage'] = (zero_count / total_size) * 100
                results['non_zero_percentage'] = 100 - results['nan_percentage'] - results['zero_percentage']
                
                # Calculate statistics from sampled values
                if all_values:
                    # Combine all sampled values
                    combined_values = np.concatenate(all_values)
                    
                    if len(combined_values) > 0:
                        results['min_value'] = float(np.min(combined_values))
                        results['max_value'] = float(np.max(combined_values))
                        results['mean_value'] = float(np.mean(combined_values))
                        results['std_value'] = float(np.std(combined_values))
            
            # Create visualizations if requested
            if create_plot and results['non_zero_percentage'] > 0:
                # Collect a sample of non-zero, non-NaN values for plotting
                plot_sample_size = min(10000, dataset.size // 100)
                plot_data = []
                
                # Sample based on dataset shape
                if len(dataset.shape) == 1:
                    indices = np.random.choice(dataset.shape[0], size=min(plot_sample_size, dataset.shape[0]), replace=False)
                    plot_data = dataset[indices]
                elif len(dataset.shape) == 2:
                    row_indices = np.random.choice(dataset.shape[0], size=min(plot_sample_size // dataset.shape[1], dataset.shape[0]), replace=False)
                    col_indices = np.random.choice(dataset.shape[1], size=min(dataset.shape[1], 5), replace=False)
                    
                    # Plot a few channels over time
                    plt.figure(figsize=(12, 8))
                    plt.title(f"fNIRS Data Sample from {os.path.basename(h5_file_path)}")
                    plt.xlabel("Time Points")
                    plt.ylabel("Value")
                    
                    for i, col in enumerate(col_indices):
                        channel_data = dataset[row_indices, col]
                        plt.plot(channel_data, label=f"Channel {col}", alpha=0.7)
                    
                    plt.legend()
                    plt.grid(True)
                    
                    # Save the plot
                    plot_path = os.path.join(os.path.dirname(h5_file_path), f"fnirs_timeseries_sample_{os.path.basename(h5_file_path)}.png")
                    plt.savefig(plot_path)
                    plt.close()
                    results['plot_path'] = plot_path
                    
                    # Sample data for histogram
                    plot_data = dataset[row_indices, :].flatten()
                elif len(dataset.shape) == 3:
                    # For 3D data, sample random frames and channels
                    frame_indices = np.random.choice(dataset.shape[0], size=min(10, dataset.shape[0]), replace=False)
                    channel_indices = np.random.choice(dataset.shape[1], size=min(5, dataset.shape[1]), replace=False)
                    
                    # Plot a few channels over time for selected frames
                    plt.figure(figsize=(12, 8))
                    plt.title(f"fNIRS Data Sample from {os.path.basename(h5_file_path)}")
                    plt.xlabel("Frames")
                    plt.ylabel("Value")
                    
                    for frame in range(min(3, len(frame_indices))):
                        for i, channel in enumerate(channel_indices[:3]):
                            channel_data = dataset[frame_indices, channel, 0].flatten()
                            plt.plot(channel_data, label=f"Frame {frame}, Channel {channel}", alpha=0.7)
                    
                    plt.legend()
                    plt.grid(True)
                    
                    # Save the plot
                    plot_path = os.path.join(os.path.dirname(h5_file_path), f"fnirs_timeseries_sample_{os.path.basename(h5_file_path)}.png")
                    plt.savefig(plot_path)
                    plt.close()
                    results['plot_path'] = plot_path
                    
                    # Sample data for histogram
                    samples = []
                    for frame in frame_indices:
                        samples.append(dataset[frame, :100, :].flatten())
                    plot_data = np.concatenate(samples)
                
                # Create histogram of values (excluding NaNs)
                if len(plot_data) > 0:
                    valid_data = plot_data[~np.isnan(plot_data)]
                    
                    if len(valid_data) > 0:
                        plt.figure(figsize=(10, 6))
                        plt.title(f"Distribution of fNIRS Values from {os.path.basename(h5_file_path)}")
                        plt.hist(valid_data, bins=100, alpha=0.7)
                        plt.xlabel("Value")
                        plt.ylabel("Frequency")
                        plt.grid(True)
                        
                        # Save the histogram
                        hist_path = os.path.join(os.path.dirname(h5_file_path), f"fnirs_histogram_{os.path.basename(h5_file_path)}.png")
                        plt.savefig(hist_path)
                        plt.close()
                        results['histogram_path'] = hist_path
    
    except Exception as e:
        logger.error(f"Error analyzing H5 file: {e}", exc_info=True)
    
    return results


def download_and_analyze_fnirs(s3_bucket, s3_key, sample_size=None, create_plot=False):
    """Download an H5 file from S3 and analyze its fNIRS data.
    
    Args:
        s3_bucket: S3 bucket name
        s3_key: S3 object key
        sample_size: If provided, analyze only a subset of the data
        create_plot: Whether to create visualization plots
        
    Returns:
        Dictionary with analysis results
    """
    # Add parent directory to path to import utils
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
    from utils.aws import init_s3_client
    
    logger.info(f"Downloading and analyzing s3://{s3_bucket}/{s3_key}")
    
    # Create temporary directory for download
    temp_dir = tempfile.mkdtemp(prefix="fnirs_analysis_")
    local_path = os.path.join(temp_dir, os.path.basename(s3_key))
    
    try:
        # Initialize S3 client
        s3_client = init_s3_client()
        
        # Download the file
        logger.info(f"Downloading to {local_path}")
        s3_client.download_file(s3_bucket, s3_key, local_path)
        logger.info(f"Downloaded {os.path.getsize(local_path)/1024/1024:.2f} MB")
        
        # Analyze the file
        results = analyze_fnirs_values(local_path, sample_size, create_plot)
        
        # Clean up temporary file unless we created plots (which would be in the same directory)
        if not create_plot:
            os.remove(local_path)
            os.rmdir(temp_dir)
        
        return results
    
    except Exception as e:
        logger.error(f"Error downloading or analyzing file: {e}", exc_info=True)
        return {
            'error': str(e),
            'file_path': f"s3://{s3_bucket}/{s3_key}"
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
    print(f"Dataset: {results['dataset_path']}")
    print(f"Shape: {results['dataset_shape']}")
    print(f"Size: {results['dataset_size_gb']:.2f} GB")
    print(f"Analyzed elements: {results['analyzed_elements']:,}")
    print("\nValue Distribution:")
    print(f"  NaN values: {results['nan_percentage']:.2f}%")
    print(f"  Zero values: {results['zero_percentage']:.2f}%")
    print(f"  Non-zero values: {results['non_zero_percentage']:.2f}%")
    
    if results['min_value'] is not None:
        print("\nStatistics (excluding NaN):")
        print(f"  Min value: {results['min_value']}")
        print(f"  Max value: {results['max_value']}")
        print(f"  Mean value: {results['mean_value']}")
        print(f"  Standard deviation: {results['std_value']}")
    
    if results['plot_path'] or results['histogram_path']:
        print("\nVisualizations:")
        if results['plot_path']:
            print(f"  Time series plot: {results['plot_path']}")
        if results['histogram_path']:
            print(f"  Histogram: {results['histogram_path']}")
    
    print("="*80)


def main():
    """Main function to run the script."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Analyze fNIRS data in H5 files")
    parser.add_argument("file_path", nargs="?", help="Path to local H5 file (if not using S3)")
    parser.add_argument("--s3", action="store_true", help="Load from S3 instead of local file")
    parser.add_argument("--bucket", default="conduit-data-dev", help="S3 bucket name")
    parser.add_argument("--key", help="S3 object key")
    parser.add_argument("--sample-size", type=int, help="Number of elements to sample (for large datasets)")
    parser.add_argument("--plot", action="store_true", help="Create visualizations")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()
    
    # Set log level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Check arguments
    if args.s3 and not args.key:
        parser.error("--s3 requires --key")
    
    if not args.s3 and not args.file_path:
        parser.error("Either provide a file_path or use --s3 with --key")
    
    # Analyze the file
    if args.s3:
        results = download_and_analyze_fnirs(args.bucket, args.key, args.sample_size, args.plot)
    else:
        results = analyze_fnirs_values(args.file_path, args.sample_size, args.plot)
    
    # Print the results
    print_analysis_summary(results)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())