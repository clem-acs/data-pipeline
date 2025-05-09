#!/usr/bin/env python3
"""
analyze-samples-and-time.py - Script to analyze sample rates in HDF5 data files

Calculates effective sample rates for each data group (EEG, fNIRS, audio, events)
by analyzing the time span and number of samples in each file.

Usage: python analyze-samples-and-time.py <folder_path>
"""

import os
import sys
import re
import h5py
import numpy as np
import pandas as pd
from tabulate import tabulate
import json
from collections import defaultdict

def natural_sort_key(s):
    """Extract timestamp from filename to sort files chronologically"""
    timestamp_match = re.search(r'_\d+_(\d+)\.h5$', s)
    if timestamp_match:
        return int(timestamp_match.group(1))
    return s

def format_rate(rate):
    """Format sample rate nicely"""
    if rate is None or np.isnan(rate):
        return "N/A"
    return f"{rate:.4f} Hz"

def extract_device_timestamps(h5file, device_type):
    """Extract timestamps for a device type from H5 file"""
    if device_type not in h5file['devices']:
        return None
        
    device_group = h5file['devices'][device_type]
    
    # First try the standard 'timestamps' dataset
    if 'timestamps' in device_group:
        return device_group['timestamps'][:]
    
    # Try frame_metadata 
    elif 'frame_metadata' in device_group:
        timestamps = []
        for ts_id in device_group['frame_metadata']:
            try:
                ts = float(ts_id)  # Group name might be timestamp
                timestamps.append(ts)
            except ValueError:
                # Check timestamp attributes
                meta_group = device_group['frame_metadata'][ts_id]
                for attr in ['server_timestamp', 'device_timestamp']:
                    if attr in meta_group.attrs:
                        timestamps.append(meta_group.attrs[attr])
                        break
        
        if timestamps:
            return np.array(timestamps)
    
    # Last resort, check individual frames
    timestamps = []
    for key in device_group.keys():
        if key in ['frame_metadata', 'frames_data']:
            continue
            
        if isinstance(device_group[key], h5py.Group):
            for attr in ['timestamp', 'device_timestamp', 'server_timestamp']:
                if attr in device_group[key].attrs:
                    timestamps.append(device_group[key].attrs[attr])
                    break
    
    if timestamps:
        return np.array(timestamps)
    
    return None

def extract_audio_timestamps(h5file):
    """Extract timestamps from audio data"""
    if 'audio' not in h5file:
        return None
        
    audio_group = h5file['audio']
    
    # Check for timestamps dataset (typically [client_ts, server_ts] pairs)
    if 'timestamps' in audio_group:
        timestamps_data = audio_group['timestamps'][:]
        
        if timestamps_data.ndim > 1 and timestamps_data.shape[1] >= 2:
            # Use client timestamps (first column)
            return timestamps_data[:, 0]
        else:
            return timestamps_data
    
    # Check for other timestamp-related datasets
    for key in audio_group.keys():
        if 'timestamp' in key.lower():
            return audio_group[key][:]
    
    return None

def extract_event_timestamps(h5file):
    """Extract timestamps from events data"""
    if 'events' not in h5file:
        return None
        
    events_group = h5file['events']
    all_timestamps = []
    
    # Iterate through event types
    for event_type in events_group:
        type_group = events_group[event_type]
        
        # Look for timestamps dataset
        if 'timestamps' in type_group:
            timestamps_data = type_group['timestamps'][:]
            
            # Check if it's a 2D array with server and client timestamps
            if timestamps_data.ndim > 1 and timestamps_data.shape[1] >= 2:
                # Use client timestamps (second column) if available
                client_timestamps = timestamps_data[:, 1]
                # Filter out zero values 
                client_timestamps = client_timestamps[client_timestamps > 0]
                
                if len(client_timestamps) > 0:
                    all_timestamps.extend(client_timestamps)
                else:
                    # Fall back to server timestamps
                    all_timestamps.extend(timestamps_data[:, 0])
            else:
                all_timestamps.extend(timestamps_data)
    
    if all_timestamps:
        return np.array(all_timestamps)
    
    return None

def analyze_h5_file(file_path):
    """Analyze a single H5 file for sample rates"""
    file_results = {
        'filename': os.path.basename(file_path),
        'data_groups': {}
    }
    
    try:
        with h5py.File(file_path, 'r') as f:
            # Process devices section (EEG, fNIRS)
            if 'devices' in f:
                for device_type in f['devices']:
                    timestamps = extract_device_timestamps(f, device_type)
                    if timestamps is not None and len(timestamps) > 0:
                        first_ts = float(np.min(timestamps))
                        last_ts = float(np.max(timestamps))
                        duration_seconds = (last_ts - first_ts) / 1000.0  # Convert ms to seconds
                        sample_count = len(timestamps)
                        
                        if duration_seconds > 0:
                            sample_rate = sample_count / duration_seconds
                        else:
                            sample_rate = None
                            
                        file_results['data_groups'][device_type] = {
                            'first_timestamp': first_ts,
                            'last_timestamp': last_ts,
                            'duration_seconds': duration_seconds,
                            'sample_count': sample_count,
                            'sample_rate': sample_rate
                        }
            
            # Process audio section
            timestamps = extract_audio_timestamps(f)
            if timestamps is not None and len(timestamps) > 0:
                first_ts = float(np.min(timestamps))
                last_ts = float(np.max(timestamps))
                duration_seconds = (last_ts - first_ts) / 1000.0  # Convert ms to seconds
                sample_count = len(timestamps)
                
                if duration_seconds > 0:
                    sample_rate = sample_count / duration_seconds
                else:
                    sample_rate = None
                    
                file_results['data_groups']['audio'] = {
                    'first_timestamp': first_ts,
                    'last_timestamp': last_ts,
                    'duration_seconds': duration_seconds,
                    'sample_count': sample_count,
                    'sample_rate': sample_rate
                }
            
            # Process events section
            timestamps = extract_event_timestamps(f)
            if timestamps is not None and len(timestamps) > 0:
                first_ts = float(np.min(timestamps))
                last_ts = float(np.max(timestamps))
                duration_seconds = (last_ts - first_ts) / 1000.0  # Convert ms to seconds
                sample_count = len(timestamps)
                
                if duration_seconds > 0:
                    sample_rate = sample_count / duration_seconds
                else:
                    sample_rate = None
                    
                file_results['data_groups']['events'] = {
                    'first_timestamp': first_ts,
                    'last_timestamp': last_ts,
                    'duration_seconds': duration_seconds,
                    'sample_count': sample_count,
                    'sample_rate': sample_rate
                }
                
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        
    return file_results

def analyze_h5_folder(folder_path):
    """Analyze all H5 files in a folder for sample rates"""
    # Find and sort H5 files chronologically
    h5_files = [f for f in os.listdir(folder_path) if f.endswith('.h5')]
    
    if not h5_files:
        print(f"No H5 files found in {folder_path}")
        return
    
    h5_files.sort(key=natural_sort_key)
    print(f"Found {len(h5_files)} H5 files in {folder_path}")
    
    # Process each file
    all_results = []
    for i, filename in enumerate(h5_files):
        file_path = os.path.join(folder_path, filename)
        print(f"Processing file {i+1}/{len(h5_files)}: {filename}")
        
        file_results = analyze_h5_file(file_path)
        all_results.append(file_results)
    
    return all_results

def print_sample_rate_summary(results):
    """Print a summary of sample rates across files"""
    if not results:
        print("No results to summarize")
        return
    
    # Collect all data groups across files
    all_data_groups = set()
    for result in results:
        all_data_groups.update(result['data_groups'].keys())
    
    # Create data for each data group
    for data_group in sorted(all_data_groups):
        print(f"\n### {data_group.upper()} Sample Rate Analysis ###")
        
        # Create table data
        table_data = []
        all_rates = []
        
        for result in results:
            if data_group in result['data_groups']:
                group_data = result['data_groups'][data_group]
                rate = group_data['sample_rate']
                if rate is not None:
                    all_rates.append(rate)
                
                table_data.append([
                    result['filename'],
                    group_data['sample_count'],
                    f"{group_data['duration_seconds']:.2f}",
                    format_rate(rate)
                ])
        
        # Print table
        print(tabulate(
            table_data,
            headers=["Filename", "Samples", "Duration (s)", "Sample Rate"],
            tablefmt="pipe"
        ))
        
        # Print statistics
        if all_rates:
            print(f"\nSample Rate Statistics:")
            print(f"  Average: {format_rate(np.mean(all_rates))}")
            print(f"  Median:  {format_rate(np.median(all_rates))}")
            print(f"  Min:     {format_rate(np.min(all_rates))}")
            print(f"  Max:     {format_rate(np.max(all_rates))}")
            print(f"  StdDev:  {format_rate(np.std(all_rates))}")

def save_results_to_json(results, folder_path):
    """Save results to a JSON file for further analysis"""
    output_path = os.path.join(folder_path, "sample_rate_analysis.json")
    
    # Convert numpy values to Python native types
    serializable_results = []
    for result in results:
        serializable_result = {
            'filename': result['filename'],
            'data_groups': {}
        }
        
        for group_name, group_data in result['data_groups'].items():
            serializable_result['data_groups'][group_name] = {
                'first_timestamp': float(group_data['first_timestamp']),
                'last_timestamp': float(group_data['last_timestamp']),
                'duration_seconds': float(group_data['duration_seconds']),
                'sample_count': int(group_data['sample_count']),
                'sample_rate': float(group_data['sample_rate']) if group_data['sample_rate'] is not None else None
            }
            
        serializable_results.append(serializable_result)
    
    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
        
    print(f"\nDetailed results saved to {output_path}")

def main():
    """Main function to parse arguments and run analysis"""
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <folder_path>")
        sys.exit(1)
    
    folder_path = sys.argv[1]
    
    if not os.path.isdir(folder_path):
        print(f"Error: {folder_path} is not a valid directory")
        sys.exit(1)
    
    # Analyze files
    results = analyze_h5_folder(folder_path)
    
    # Print summary
    if results:
        print_sample_rate_summary(results)
        save_results_to_json(results, folder_path)

if __name__ == "__main__":
    try:
        # Check if pandas and tabulate are installed
        import pandas as pd
        import tabulate as tabulate_module  # Import with different name to avoid conflict
    except ImportError:
        print("This script requires pandas and tabulate packages.")
        print("Please install them with: pip install pandas tabulate")
        sys.exit(1)
        
    main()