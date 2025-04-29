#!/usr/bin/env python
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def visualize_snirf_data():
    """Visualize key aspects of the SNIRF data files"""
    snirf_files = list(Path('.').glob('*.snirf'))
    file_names = [f.name for f in snirf_files]
    
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(15, 12))
    
    # 1. Time Series Plot - Compare first channel from each file
    ax1 = fig.add_subplot(3, 1, 1)
    
    time_data = None
    sample_data = {}
    
    for i, snirf_file in enumerate(snirf_files):
        with h5py.File(snirf_file, 'r') as f:
            # Get time data (should be the same across files)
            if time_data is None:
                time_data = f['nirs/data1/time'][:]
            
            # Get the first channel of data
            data = f['nirs/data1/dataTimeSeries'][:, 0]
            
            # Store for later use
            sample_data[snirf_file.name] = data
            
            # Plot only first 500 time points for clarity
            ax1.plot(time_data[:500], data[:500], label=snirf_file.name)
    
    ax1.set_title('First Channel Time Series (First 500 Points)')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.legend()
    ax1.grid(True)
    
    # 2. Histograms - Distribution of values in each file
    ax2 = fig.add_subplot(3, 1, 2)
    
    for name, data in sample_data.items():
        # Take a random sample to avoid excessive computation
        sample = np.random.choice(data, size=min(1000, len(data)), replace=False)
        
        # Plot histogram - use log scale due to large differences in data ranges
        ax2.hist(sample, bins=50, alpha=0.5, label=name)
    
    ax2.set_title('Distribution of Values (First Channel)')
    ax2.set_xlabel('Value')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    ax2.grid(True)
    
    # 3. Data Correlation Matrix
    ax3 = fig.add_subplot(3, 1, 3)
    
    # Normalize each dataset for fair comparison
    normalized_data = {}
    for name, data in sample_data.items():
        # Z-score normalization
        normalized_data[name] = (data - np.mean(data)) / np.std(data)
    
    # Create correlation matrix for the first 500 time points
    time_points = 500
    correlation_matrix = np.zeros((len(file_names), len(file_names)))
    
    for i, name1 in enumerate(file_names):
        for j, name2 in enumerate(file_names):
            # Calculate correlation coefficient
            corr = np.corrcoef(
                normalized_data[name1][:time_points],
                normalized_data[name2][:time_points]
            )[0, 1]
            correlation_matrix[i, j] = corr
    
    # Plot correlation matrix
    im = ax3.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax3)
    
    # Add labels
    ax3.set_title('Correlation between Normalized First Channels')
    ax3.set_xticks(np.arange(len(file_names)))
    ax3.set_yticks(np.arange(len(file_names)))
    ax3.set_xticklabels(file_names, rotation=90)
    ax3.set_yticklabels(file_names)
    
    # Add correlation values to the heatmap
    for i in range(len(file_names)):
        for j in range(len(file_names)):
            text = ax3.text(j, i, f"{correlation_matrix[i, j]:.2f}",
                          ha="center", va="center", color="black")
    
    plt.tight_layout()
    plt.savefig('snirf_data_visualization.png')
    print("Visualization saved as 'snirf_data_visualization.png'")
    
    # Additional analysis: Compare the structures of different dataTypes
    print("\n===== DATA TYPE STRUCTURE COMPARISON =====")
    
    # Extract information about the different data types
    for snirf_file in snirf_files:
        print(f"\n{snirf_file.name}:")
        
        with h5py.File(snirf_file, 'r') as f:
            # Get first measurement list to identify data type
            measurement_key = next((k for k in f['nirs/data1'].keys() if k.startswith('measurementList')), None)
            
            if measurement_key:
                measurement = f[f'nirs/data1/{measurement_key}']
                if 'dataType' in measurement:
                    data_type = measurement['dataType'][()]
                    if 'dataTypeLabel' in measurement:
                        data_type_label = measurement['dataTypeLabel'][()]
                        if isinstance(data_type_label, bytes):
                            data_type_label = data_type_label.decode('utf-8')
                        print(f"  Data Type: {data_type} - {data_type_label}")
                    else:
                        print(f"  Data Type: {data_type}")
            
            # Get the size and shape of the data
            data_time_series = f['nirs/data1/dataTimeSeries']
            print(f"  Data Shape: {data_time_series.shape}")
            print(f"  Data Size (MB): {data_time_series.size * data_time_series.dtype.itemsize / (1024 * 1024):.2f}")
            
            # Calculate basic statistics
            print(f"  Data Range: {np.min(data_time_series[:, 0]):g} to {np.max(data_time_series[:, 0]):g}")
            print(f"  Data Mean: {np.mean(data_time_series[:, 0]):g}")
            print(f"  Data Std Dev: {np.std(data_time_series[:, 0]):g}")
            
            # Check for patterns in the data structure
            num_channels = data_time_series.shape[1]
            num_sources = len(set(f[f'nirs/data1/{meas}']['sourceIndex'][()] 
                                for meas in f['nirs/data1'].keys() 
                                if meas.startswith('measurementList')))
            num_detectors = len(set(f[f'nirs/data1/{meas}']['detectorIndex'][()] 
                                  for meas in f['nirs/data1'].keys() 
                                  if meas.startswith('measurementList')))
            
            print(f"  Number of Channels: {num_channels}")
            print(f"  Unique Sources: {num_sources}")
            print(f"  Unique Detectors: {num_detectors}")
            
            # Check wavelength information if available
            if 'nirs/probe/wavelengths' in f:
                wavelengths = f['nirs/probe/wavelengths'][:]
                print(f"  Wavelengths: {wavelengths}")

if __name__ == "__main__":
    visualize_snirf_data()