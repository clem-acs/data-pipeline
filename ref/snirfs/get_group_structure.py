#!/usr/bin/env python3
"""
Script to print the complete group structure of SNIRF/HDF5 files.
"""
import os
import sys
import h5py

def print_group_structure(name, obj):
    """Print information about a group or dataset."""
    indent = '  ' * name.count('/')
    
    if isinstance(obj, h5py.Group):
        print(f"{indent}GROUP: {name}")
    elif isinstance(obj, h5py.Dataset):
        shape_str = str(obj.shape) if obj.shape else "scalar"
        dtype_str = str(obj.dtype)
        print(f"{indent}DATASET: {name} (Shape: {shape_str}, Type: {dtype_str})")
    else:
        print(f"{indent}OTHER: {name}")

def list_group_contents(group, prefix=''):
    """List all keys in a group."""
    for key in group.keys():
        if prefix:
            full_key = f"{prefix}/{key}"
        else:
            full_key = key
        
        print(f"Key: {full_key}")
        
        item = group[key]
        if isinstance(item, h5py.Group):
            list_group_contents(item, full_key)

def get_all_groups(file_path):
    """Get full structure of HDF5/SNIRF file."""
    try:
        with h5py.File(file_path, 'r') as f:
            print(f"\nFile: {os.path.basename(file_path)}")
            print("\n=== Complete Group Structure ===")
            f.visititems(print_group_structure)
            
            print("\n=== Root Level Keys ===")
            for key in f.keys():
                print(f"- {key}")
            
            if 'nirs' in f:
                print("\n=== NIRS Group Contents ===")
                nirs_keys = list(f['nirs'].keys())
                for key in nirs_keys:
                    print(f"- {key}")
                
                # Count the different types of groups
                data_groups = [k for k in nirs_keys if k.startswith('data')]
                aux_groups = [k for k in nirs_keys if k.startswith('aux')]
                stim_groups = [k for k in nirs_keys if k.startswith('stim')]
                
                print(f"\nSummary:")
                print(f"- Data groups: {len(data_groups)}")
                print(f"- Auxiliary data groups: {len(aux_groups)}")
                print(f"- Stimulus groups: {len(stim_groups)}")
                print(f"- Other groups: {len(nirs_keys) - len(data_groups) - len(aux_groups) - len(stim_groups)}")
                
                if 'data1' in f['nirs']:
                    data1 = f['nirs']['data1']
                    print("\n=== Measurement Lists in data1 ===")
                    ml_keys = [k for k in data1.keys() if k.startswith('measurementList')]
                    print(f"Total measurement lists: {len(ml_keys)}")
                    
                    # Get a sample of the measurement lists
                    sample_size = min(5, len(ml_keys))
                    if sample_size > 0:
                        print(f"\nSample of first {sample_size} measurement lists:")
                        for i, ml_key in enumerate(sorted(ml_keys)[:sample_size]):
                            print(f"- {ml_key}")
    except Exception as e:
        print(f"Error: {str(e)}")

def main():
    """Process command line arguments."""
    if len(sys.argv) < 2:
        print("Usage: python get_group_structure.py <snirf_file_path>")
        return
    
    file_path = sys.argv[1]
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found")
        return
    
    get_all_groups(file_path)

if __name__ == "__main__":
    main()