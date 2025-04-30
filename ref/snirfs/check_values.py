#!/usr/bin/env python3
"""
Check for invalid or non-positive values in SNIRF data.
"""
import h5py
import numpy as np
import sys
import os

def check_data_values(file_path):
    """Check for any invalid or non-positive values in SNIRF data."""
    try:
        print(f"Checking values in: {os.path.basename(file_path)}")
        with h5py.File(file_path, 'r') as f:
            if 'nirs' not in f:
                print("Error: No 'nirs' group found in the file.")
                return
                
            nirs = f['nirs']
            
            # Find data elements
            data_elements = [key for key in nirs.keys() if key.startswith('data')]
            if not data_elements:
                print("Error: No data elements found.")
                return
                
            for data_key in data_elements:
                data_element = nirs[data_key]
                print(f"\nChecking {data_key}:")
                
                if 'dataTimeSeries' not in data_element:
                    print(f"  No dataTimeSeries found in {data_key}")
                    continue
                    
                data = data_element['dataTimeSeries'][:]
                shape = data.shape
                print(f"  Data shape: {shape}")
                
                # Check for various problematic values
                nan_count = np.isnan(data).sum()
                inf_count = np.isinf(data).sum()
                neg_inf_count = np.isneginf(data).sum()
                zero_count = (data == 0).sum()
                negative_count = np.sum(data < 0) - neg_inf_count  # Exclude -inf in negative count
                
                total_elements = data.size
                problematic_elements = nan_count + inf_count + zero_count + negative_count
                
                print(f"  Total data points: {total_elements:,}")
                print(f"  NaN values: {nan_count:,} ({nan_count/total_elements*100:.5f}%)")
                print(f"  Inf values: {inf_count:,} ({inf_count/total_elements*100:.5f}%)")
                print(f"  -Inf values: {neg_inf_count:,} ({neg_inf_count/total_elements*100:.5f}%)")
                print(f"  Zero values: {zero_count:,} ({zero_count/total_elements*100:.5f}%)")
                print(f"  Negative values (excluding -inf): {negative_count:,} ({negative_count/total_elements*100:.5f}%)")
                print(f"  Total problematic values: {problematic_elements:,} ({problematic_elements/total_elements*100:.5f}%)")
                
                # Calculate percentiles for valid values
                valid_data = data[np.isfinite(data) & (data > 0)]
                if len(valid_data) > 0:
                    percentiles = np.percentile(valid_data, [0, 0.1, 1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9, 100])
                    print("\n  Percentiles of valid positive values:")
                    print(f"    Min (0th): {percentiles[0]:.2f}")
                    print(f"    0.1th: {percentiles[1]:.2f}")
                    print(f"    1st: {percentiles[2]:.2f}")
                    print(f"    5th: {percentiles[3]:.2f}")
                    print(f"    10th: {percentiles[4]:.2f}")
                    print(f"    25th: {percentiles[5]:.2f}")
                    print(f"    Median (50th): {percentiles[6]:.2f}")
                    print(f"    75th: {percentiles[7]:.2f}")
                    print(f"    90th: {percentiles[8]:.2f}")
                    print(f"    95th: {percentiles[9]:.2f}")
                    print(f"    99th: {percentiles[10]:.2f}")
                    print(f"    99.9th: {percentiles[11]:.2f}")
                    print(f"    Max (100th): {percentiles[12]:.2f}")
                
                # If there are any non-positive values, check their distribution across channels
                if zero_count > 0 or negative_count > 0 or nan_count > 0 or inf_count > 0:
                    print("\n  Checking distribution of problematic values across channels:")
                    
                    channels_with_zeros = np.sum(np.any(data == 0, axis=0))
                    channels_with_negative = np.sum(np.any(data < 0, axis=0))
                    channels_with_nan = np.sum(np.any(np.isnan(data), axis=0))
                    channels_with_inf = np.sum(np.any(np.isinf(data), axis=0))
                    
                    print(f"    Channels with zero values: {channels_with_zeros} ({channels_with_zeros/shape[1]*100:.2f}%)")
                    print(f"    Channels with negative values: {channels_with_negative} ({channels_with_negative/shape[1]*100:.2f}%)")
                    print(f"    Channels with NaN values: {channels_with_nan} ({channels_with_nan/shape[1]*100:.2f}%)")
                    print(f"    Channels with Inf values: {channels_with_inf} ({channels_with_inf/shape[1]*100:.2f}%)")
                    
                    # Check if there are any entirely invalid channels (all values are problematic)
                    all_zeros_channels = np.sum(np.all(data == 0, axis=0))
                    all_negative_channels = np.sum(np.all(data < 0, axis=0))
                    all_nan_channels = np.sum(np.all(np.isnan(data), axis=0))
                    all_inf_channels = np.sum(np.all(np.isinf(data), axis=0))
                    
                    print(f"    Channels with ALL zero values: {all_zeros_channels}")
                    print(f"    Channels with ALL negative values: {all_negative_channels}")
                    print(f"    Channels with ALL NaN values: {all_nan_channels}")
                    print(f"    Channels with ALL Inf values: {all_inf_channels}")
                
    except Exception as e:
        print(f"Error analyzing file: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_values.py <path/to/snirf_file>")
        sys.exit(1)
        
    file_path = sys.argv[1]
    check_data_values(file_path)