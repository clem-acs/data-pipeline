#!/usr/bin/env python3
"""
Script to print THINKING_START and THINKING_STOP events from an HDF5 file.
Usage: python print-thinking-events.py /path/to/file.h5
"""

import sys
import h5py
import json
import numpy as np

def print_event_dataset(h5_file, event_type):
    """Print the full dataset for a given event type."""
    print(f"\n===== {event_type.upper()} EVENTS =====")
    
    if 'events' not in h5_file:
        print(f"No events group found in the H5 file")
        return
    
    events_group = h5_file['events']
    
    if event_type not in events_group:
        print(f"No {event_type} events found in the H5 file")
        return
    
    event_group = events_group[event_type]
    
    # Print dataset info
    print(f"Dataset shape and size information:")
    for dataset_name in ['data', 'timestamps', 'event_ids']:
        if dataset_name in event_group:
            dataset = event_group[dataset_name]
            print(f"  {dataset_name}: shape={dataset.shape}, len={len(dataset)}")
        else:
            print(f"  {dataset_name}: Not found")
    
    # Print all events
    if 'data' in event_group and 'timestamps' in event_group and 'event_ids' in event_group:
        data = event_group['data']
        timestamps = event_group['timestamps']
        event_ids = event_group['event_ids']
        
        print(f"\nPrinting all {len(data)} events:")
        for i in range(len(data)):
            event_data = data[i]
            # Try to parse JSON if it's a string
            if isinstance(event_data, (bytes, str)):
                try:
                    event_data = json.loads(event_data)
                except:
                    pass
            
            # Get timestamps - timestamps array has shape (n, 2) where index 0 is server_ts, index 1 is client_ts
            server_ts = timestamps[i][0] if timestamps[i].size > 0 else None
            client_ts = timestamps[i][1] if timestamps[i].size > 1 else None
            
            print(f"\nEvent {i+1}:")
            print(f"  ID: {event_ids[i]}")
            print(f"  Server timestamp: {server_ts}")
            print(f"  Client timestamp: {client_ts}")
            print(f"  Data: {event_data}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python print-thinking-events.py /path/to/file.h5")
        sys.exit(1)
    
    h5_path = sys.argv[1]
    
    try:
        with h5py.File(h5_path, 'r') as f:
            # Print summary of H5 file
            print(f"H5 File: {h5_path}")
            print("Available groups:", list(f.keys()))
            
            if 'events' in f:
                print("Available event types:", list(f['events'].keys()))
            
            # Print thinking events
            print_event_dataset(f, 'thinking_start')
            print_event_dataset(f, 'thinking_stop')
    
    except Exception as e:
        print(f"Error opening or reading H5 file: {e}")

if __name__ == "__main__":
    main()