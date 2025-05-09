#!/usr/bin/env python3
"""
HDF5 Inspector - A simple script to explore and print the contents of HDF5 files
"""

import h5py
import numpy as np
import argparse
import os

def print_separator(length=60):
    """Print a separator line."""
    print("-" * length)

def inspect_hdf5(file_path):
    """Inspect and print the contents of an HDF5 file."""
    print(f"Inspecting HDF5 file: {file_path}")
    print_separator()
    
    with h5py.File(file_path, 'r') as f:
        # Print file attributes
        print("FILE ATTRIBUTES:")
        for key, value in f.attrs.items():
            print(f"  {key}: {value}")
        print_separator()
        
        # Print groups
        print("FILE STRUCTURE:")
        
        def print_group(name, obj):
            indent = "  " * name.count('/')
            if isinstance(obj, h5py.Group):
                print(f"{indent}Group: {name}/")
            elif isinstance(obj, h5py.Dataset):
                shape_str = " x ".join(str(dim) for dim in obj.shape)
                print(f"{indent}Dataset: {name} (Shape: {shape_str}, Type: {obj.dtype})")
        
        f.visititems(print_group)
        print_separator()
        
        # Print detailed info for each group
        def explore_group(group, path=""):
            """Recursively explore groups and print details."""
            print(f"GROUP: {path or '/'}")
            
            # Print group attributes
            if len(group.attrs) > 0:
                print("  Attributes:")
                for key, value in group.attrs.items():
                    print(f"    {key}: {value}")
            
            # Iterate through items in the group
            for key, item in group.items():
                item_path = f"{path}/{key}" if path else key
                
                if isinstance(item, h5py.Group):
                    # Recursively explore subgroups
                    print_separator()
                    explore_group(item, item_path)
                elif isinstance(item, h5py.Dataset):
                    # Print dataset details
                    print(f"  DATASET: {key}")
                    print(f"    Shape: {item.shape}")
                    print(f"    Type: {item.dtype}")
                    
                    # Print dataset attributes
                    if len(item.attrs) > 0:
                        print("    Attributes:")
                        for attr_key, attr_value in item.attrs.items():
                            print(f"      {attr_key}: {attr_value}")
                    
                    # Print sample data for the dataset
                    print("    Sample Data:")
                    if len(item.shape) == 1:
                        # For 1D data, print first few elements
                        data_sample = item[:min(5, item.shape[0])]
                        print(f"      First {len(data_sample)} elements: {data_sample}")
                    elif len(item.shape) == 2:
                        # For 2D data, print a small corner
                        rows = min(3, item.shape[0])
                        cols = min(3, item.shape[1])
                        print(f"      Top-left corner ({rows}x{cols}):")
                        for i in range(rows):
                            print(f"        {item[i, :cols]}")
                    else:
                        # For higher dimensions, just show stats
                        print(f"      [Data has {len(item.shape)} dimensions]")
                    
                    # Print basic statistics for numerical data
                    if np.issubdtype(item.dtype, np.number):
                        try:
                            data_sample = item[:]
                            print("    Statistics:")
                            print(f"      Min: {np.min(data_sample):.4f}")
                            print(f"      Max: {np.max(data_sample):.4f}")
                            print(f"      Mean: {np.mean(data_sample):.4f}")
                            print(f"      Std Dev: {np.std(data_sample):.4f}")
                        except Exception as e:
                            print(f"      [Error computing statistics: {e}]")
        
        # Start exploring from the root group
        explore_group(f)

def main():
    parser = argparse.ArgumentParser(description="Inspect and print HDF5 file contents")
    parser.add_argument("file", help="Path to the HDF5 file to inspect")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.file):
        print(f"Error: File {args.file} not found")
        return
    
    inspect_hdf5(args.file)

if __name__ == "__main__":
    main()