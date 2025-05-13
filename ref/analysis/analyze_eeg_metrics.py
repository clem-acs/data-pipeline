#!/usr/bin/env python
"""
Script to analyze EEG metrics over time for EEG data files.
Computes metrics for each 2-second window and plots the results.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sys
import glob
import logging

# Import the metrics module from your codebase
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))
from utils.eeg_metrics import calculate_channel_metrics, calculate_all_channels_metrics

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('analyze_eeg_metrics')

def compute_metrics_over_time(eeg_data, sampling_rate=500, window_size_seconds=2, channels_to_analyze=None):
    """
    Compute EEG metrics for each time window.
    
    Args:
        eeg_data: EEG data array with shape (channels, samples)
        sampling_rate: Sampling rate in Hz
        window_size_seconds: Size of each window in seconds
        channels_to_analyze: List of channel indices to analyze (None for all)
        
    Returns:
        Dictionary of metrics over time
    """
    num_channels, num_samples = eeg_data.shape
    window_size = int(window_size_seconds * sampling_rate)
    num_windows = num_samples // window_size
    
    # Use all channels if none specified
    if channels_to_analyze is None:
        channels_to_analyze = list(range(num_channels))
    
    # Initialize metrics storage
    metrics_over_time = {
        'mean': np.zeros((len(channels_to_analyze), num_windows)),
        'std_dev': np.zeros((len(channels_to_analyze), num_windows)),
        'kurtosis': np.zeros((len(channels_to_analyze), num_windows)),
        'line_noise_ratio_50hz': np.zeros((len(channels_to_analyze), num_windows)),
        'line_noise_ratio_60hz': np.zeros((len(channels_to_analyze), num_windows))
    }
    
    # Process each time window
    for win_idx in range(num_windows):
        start_idx = win_idx * window_size
        end_idx = start_idx + window_size
        
        # Extract window data for all channels
        window_data = eeg_data[:, start_idx:end_idx]
        
        # Calculate metrics for all specified channels in this window
        metrics_dict = calculate_all_channels_metrics(
            window_data, 
            sampling_rate=sampling_rate,
            channel_indices=channels_to_analyze
        )
        
        # Store metrics from each channel into our time series arrays
        for i, ch_idx in enumerate(channels_to_analyze):
            channel_name = f"channel_{ch_idx}"
            if channel_name in metrics_dict:
                for metric_name in metrics_over_time.keys():
                    metrics_over_time[metric_name][i, win_idx] = metrics_dict[channel_name][metric_name]
    
    return metrics_over_time, channels_to_analyze

def plot_metrics(metrics_over_time, channels, file_name, sampling_rate=500, window_size_seconds=2):
    """
    Plot metrics over time for selected channels.
    
    Args:
        metrics_over_time: Dictionary of metrics arrays
        channels: List of channel indices that were analyzed
        file_name: Name of the file for the plot title
        sampling_rate: Sampling rate in Hz
        window_size_seconds: Size of each window in seconds
    """
    num_windows = metrics_over_time['mean'].shape[1]
    
    # Time axis in seconds
    time_axis = np.arange(num_windows) * window_size_seconds
    
    # Set up the figure with 5 subplots (one for each metric)
    plt.figure(figsize=(15, 12))
    gs = GridSpec(5, 1, figure=plt.gcf(), hspace=0.3)
    
    # Plot titles and data
    metric_titles = {
        'mean': 'Mean Value (mV)',
        'std_dev': 'Standard Deviation (mV)',
        'kurtosis': 'Kurtosis',
        'line_noise_ratio_50hz': 'Line Noise Ratio (50Hz)',
        'line_noise_ratio_60hz': 'Line Noise Ratio (60Hz)'
    }
    
    # Use different colors for each channel
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    for i, (metric_name, title) in enumerate(metric_titles.items()):
        ax = plt.subplot(gs[i])
        
        for j, ch_idx in enumerate(channels):
            # Plot data for this channel
            color_idx = j % len(color_cycle)  # Cycle through colors if more channels than colors
            ax.plot(time_axis, metrics_over_time[metric_name][j, :], 
                    label=f'Channel {ch_idx}', 
                    linewidth=1.5,
                    color=color_cycle[color_idx])
        
        # Add labels and legend
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        # Add legend to the first plot only
        if i == 0:
            ax.legend(loc='upper right')
    
    # Add overall title
    plt.suptitle(f'EEG Metrics Over Time: {os.path.basename(file_name)}', fontsize=16)
    
    # Save and show the plot
    output_file = f"{os.path.splitext(file_name)[0]}_metrics.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    plt.show()

def find_interesting_channels(eeg_data, sampling_rate=500, num_channels=6):
    """
    Find interesting channels to analyze based on variance or other properties.
    
    Args:
        eeg_data: EEG data array
        sampling_rate: Sampling rate in Hz
        num_channels: Number of channels to select
        
    Returns:
        List of channel indices
    """
    # Calculate standard deviation for each channel
    channel_stds = np.std(eeg_data, axis=1)
    
    # Get channels with highest standard deviation
    high_std_channels = np.argsort(channel_stds)[-num_channels//2:]
    
    # Get channels with median standard deviation
    median_std_channels = np.argsort(channel_stds)[len(channel_stds)//2-num_channels//4:len(channel_stds)//2+num_channels//4]
    
    # Combine high and median channels 
    interesting_channels = np.concatenate([high_std_channels, median_std_channels])
    
    # Add a couple of low std channels
    low_std_channels = np.argsort(channel_stds)[:num_channels-len(interesting_channels)]
    interesting_channels = np.concatenate([interesting_channels, low_std_channels])
    
    # Ensure we have exactly the requested number
    interesting_channels = interesting_channels[:num_channels]
    
    # Sort for display purposes
    interesting_channels.sort()
    
    return interesting_channels.tolist()

def analyze_file(file_path, sampling_rate=500, window_size_seconds=2, specific_channels=None):
    """Analyze a single EEG file and plot metrics."""
    try:
        # Load the EEG data
        eeg_data = np.load(file_path)
        logger.info(f"Loaded file: {file_path}, shape: {eeg_data.shape}")
        
        # Find interesting channels if not specified
        channels_to_analyze = specific_channels
        if channels_to_analyze is None:
            channels_to_analyze = find_interesting_channels(eeg_data, sampling_rate)
            logger.info(f"Selected interesting channels: {channels_to_analyze}")
        
        # Compute metrics over time
        metrics, analyzed_channels = compute_metrics_over_time(
            eeg_data, 
            sampling_rate, 
            window_size_seconds,
            channels_to_analyze
        )
        logger.info(f"Computed metrics over {metrics['mean'].shape[1]} windows")
        
        # Plot metrics
        plot_metrics(metrics, analyzed_channels, file_path, sampling_rate, window_size_seconds)
        
        return True
    except Exception as e:
        logger.error(f"Error analyzing file {file_path}: {e}")
        return False

def main():
    """Main function to analyze EEG files."""
    data_dir = os.path.join(os.path.expanduser("~"), "Desktop", "data")
    
    if not os.path.exists(data_dir):
        logger.error(f"Directory {data_dir} does not exist")
        return
    
    # Get all .npy files in directory
    npy_files = glob.glob(os.path.join(data_dir, "*.npy"))
    
    if not npy_files:
        logger.error(f"No .npy files found in {data_dir}")
        return
    
    # Sort files by size and choose a medium-sized file
    npy_files.sort(key=os.path.getsize)
    sample_file = npy_files[len(npy_files) // 2]
    
    logger.info(f"Analyzing sample file: {sample_file}")
    
    # Determine sampling rate (if possible)
    sampling_rate = 500  # Based on the data description
    logger.info(f"Using sampling rate: {sampling_rate} Hz")
    
    # Analyze the file
    specific_channels = [0, 5, 10, 20, 30, 40, 50, 58]  # Example: specific channels to analyze
    analyze_file(sample_file, sampling_rate, specific_channels=specific_channels)

if __name__ == "__main__":
    main()