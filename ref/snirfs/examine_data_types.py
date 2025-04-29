#!/usr/bin/env python
import h5py
import os
import numpy as np
from pathlib import Path
import pandas as pd

def extract_data_types(snirf_file):
    """Extract data type information from a SNIRF file"""
    data_type_info = []
    
    with h5py.File(snirf_file, 'r') as f:
        file_name = os.path.basename(snirf_file)
        i = 1
        while f'nirs/data{i}' in f:
            data_group = f[f'nirs/data{i}']
            
            for key in data_group.keys():
                if key.startswith('measurementList'):
                    measurement = {}
                    measurement['file_name'] = file_name
                    measurement['data_element'] = i
                    measurement['measurement_key'] = key
                    
                    # Extract all attributes
                    for attr_key in data_group[key].keys():
                        value = data_group[key][attr_key][()]
                        if isinstance(value, bytes):
                            value = value.decode('utf-8')
                        measurement[attr_key] = value
                    
                    data_type_info.append(measurement)
            i += 1
    
    return data_type_info

def analyze_data_types():
    """Analyze data types across all SNIRF files"""
    snirf_files = list(Path('.').glob('*.snirf'))
    print(f"Found {len(snirf_files)} SNIRF files")
    
    all_measurements = []
    
    for snirf_file in snirf_files:
        print(f"Processing {snirf_file}...")
        measurements = extract_data_types(snirf_file)
        # Only take a sample of up to 100 measurements to avoid excessive output
        sample_size = min(100, len(measurements))
        sampled_measurements = measurements[:sample_size]
        all_measurements.extend(sampled_measurements)
        print(f"  Extracted {len(measurements)} measurements, sampled {sample_size}")
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(all_measurements)
    
    # Print summary of data types
    print("\n===== DATA TYPE SUMMARY =====")
    data_type_groups = df.groupby(['file_name', 'dataType'])
    
    for (file_name, data_type), group in data_type_groups:
        print(f"\n{file_name} - Data Type {data_type}:")
        
        # Count by data type label if available
        if 'dataTypeLabel' in group.columns:
            print(f"  Label: {group['dataTypeLabel'].iloc[0]}")
        
        # Sample of the first measurement
        sample_row = group.iloc[0]
        print(f"  Sample measurement:")
        for column in sample_row.index:
            if column not in ['file_name', 'data_element', 'measurement_key']:
                print(f"    {column}: {sample_row[column]}")
        
        print(f"  Count: {len(group)}")
    
    # Quick look at time information
    if 'nirs/data1/time' in h5py.File(snirf_files[0], 'r'):
        with h5py.File(snirf_files[0], 'r') as f:
            time_data = f['nirs/data1/time'][:]
            print("\n===== TIME DATA =====")
            print(f"Time points: {len(time_data)}")
            print(f"Time range: {time_data[0]} to {time_data[-1]} seconds")
            print(f"Time step: {time_data[1] - time_data[0]} seconds")
            print(f"Sample rate: {1/(time_data[1] - time_data[0])} Hz")
    
    # Examine data shape and values for each type
    print("\n===== DATA SAMPLES =====")
    for snirf_file in snirf_files:
        file_name = os.path.basename(snirf_file)
        print(f"\n{file_name}:")
        
        with h5py.File(snirf_file, 'r') as f:
            data = f['nirs/data1/dataTimeSeries']
            shape = data.shape
            print(f"  Data shape: {shape}")
            
            # Sample a few columns from the data
            sample_cols = min(5, shape[1])
            sample_rows = min(3, shape[0])
            
            print(f"  Sample data (first {sample_rows} rows, first {sample_cols} columns):")
            for row in range(sample_rows):
                values = data[row, :sample_cols]
                print(f"    Row {row}: {values}")
            
            # Stats about the data
            print(f"  Data statistics:")
            try:
                sample_data = data[:1000, :100]  # Take a manageable sample
                print(f"    Min: {np.min(sample_data)}")
                print(f"    Max: {np.max(sample_data)}")
                print(f"    Mean: {np.mean(sample_data)}")
                print(f"    Std Dev: {np.std(sample_data)}")
                print(f"    NaN count: {np.isnan(sample_data).sum()}")
                print(f"    Zero count: {(sample_data == 0).sum()}")
            except Exception as e:
                print(f"    Error calculating statistics: {e}")

if __name__ == "__main__":
    analyze_data_types()