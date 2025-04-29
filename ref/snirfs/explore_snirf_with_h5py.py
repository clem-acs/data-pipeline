#!/usr/bin/env python
import h5py
import os
import numpy as np
from pathlib import Path

def explore_h5_structure(h5file, prefix=''):
    """Recursively explore and print the structure of an HDF5 file"""
    for key in h5file.keys():
        item = h5file[key]
        path = f"{prefix}/{key}" if prefix else key
        
        if isinstance(item, h5py.Group):
            print(f"GROUP: {path}")
            explore_h5_structure(item, path)
        elif isinstance(item, h5py.Dataset):
            data_shape = item.shape
            data_type = item.dtype
            
            # Print basic info
            print(f"DATASET: {path}")
            print(f"  Shape: {data_shape}")
            print(f"  Type: {data_type}")
            
            # If it's a small dataset or 1D array with reasonable size, show sample values
            if len(data_shape) == 0 or (np.prod(data_shape) <= 10 and len(data_shape) <= 2):
                try:
                    data = item[()]
                    if isinstance(data, bytes):
                        data = data.decode('utf-8', errors='replace')
                    print(f"  Value: {data}")
                except Exception as e:
                    print(f"  Error reading value: {e}")
            # For arrays, show a few sample values
            elif len(data_shape) >= 1:
                try:
                    if len(data_shape) == 1 and data_shape[0] > 0:
                        sample = item[0:min(5, data_shape[0])]
                    elif len(data_shape) == 2 and data_shape[0] > 0 and data_shape[1] > 0:
                        sample = item[0:min(3, data_shape[0]), 0:min(3, data_shape[1])]
                    else:
                        # For higher dimensions, just grab the first few elements
                        sample = item[tuple(slice(0, min(2, s)) for s in data_shape)]
                    
                    print(f"  Sample data: {sample}")
                except Exception as e:
                    print(f"  Error reading sample: {e}")
        else:
            print(f"OTHER: {path}")

def main():
    # Find all SNIRF files in the current directory
    snirf_files = list(Path('.').glob('*.snirf'))
    print(f"Found {len(snirf_files)} SNIRF files")
    
    for snirf_file in snirf_files:
        print(f"\n\n{'='*80}")
        print(f"EXPLORING: {snirf_file}")
        print(f"{'='*80}")
        
        try:
            with h5py.File(snirf_file, 'r') as f:
                explore_h5_structure(f)
        except Exception as e:
            print(f"Error opening {snirf_file}: {e}")

if __name__ == "__main__":
    main()