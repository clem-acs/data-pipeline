#!/usr/bin/env python3
"""
get-base-metadata.py - Simple script to print the base metadata from an HDF5 file

Usage:
    python get-base-metadata.py --file /path/to/file.h5
"""

import argparse
import h5py
import json
import sys


def print_metadata(file_path):
    """Print the metadata group contents from an HDF5 file."""
    try:
        with h5py.File(file_path, 'r') as f:
            print(f"HDF5 File: {file_path}")
            print("\n=== Base Metadata ===\n")
            
            # Check if metadata group exists
            if 'metadata' in f:
                metadata_group = f['metadata']
                
                # Print all attributes
                print("Attributes:")
                for key, value in metadata_group.attrs.items():
                    # Try to parse JSON strings
                    if key.endswith('_json'):
                        try:
                            parsed_value = json.loads(value)
                            print(f"  {key[:-5]} (JSON): {json.dumps(parsed_value, indent=2)}")
                        except:
                            print(f"  {key}: {value}")
                    else:
                        print(f"  {key}: {value}")
                
                # Check if there are any datasets or subgroups
                if len(metadata_group) > 0:
                    print("\nSubgroups and Datasets:")
                    for name in metadata_group:
                        item = metadata_group[name]
                        if isinstance(item, h5py.Group):
                            print(f"  Group: {name}")
                        else:  # Dataset
                            print(f"  Dataset: {name}, Shape: {item.shape}, Type: {item.dtype}")
            else:
                print("No metadata group found in the file.")
                
                # List root level groups as an alternative
                print("\nRoot level groups:")
                for name in f:
                    print(f"  {name}")
    except Exception as e:
        print(f"Error opening or reading HDF5 file: {str(e)}", file=sys.stderr)
        return False
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Print base metadata from an HDF5 file")
    parser.add_argument("--file", required=True, help="Path to the HDF5 file")
    
    args = parser.parse_args()
    
    success = print_metadata(args.file)
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()