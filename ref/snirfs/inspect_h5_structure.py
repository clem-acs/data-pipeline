#!/usr/bin/env python3
import h5py
import numpy as np
import sys

def inspect_h5_structure(file_path):
    """Inspect the structure of an HDF5 file."""
    try:
        with h5py.File(file_path, 'r') as f:
            print(f"Inspecting file: {file_path}")
            print("\nGroups in file:")
            for key in f.keys():
                print(f"- {key}")
            
            if 'nirs' in f:
                print("\nDetails of nirs group:")
                nirs = f['nirs']
                for nirs_key in nirs.keys():
                    print(f"- {nirs_key}")
                
                # The structure is different - nirs doesn't have numbered items
                # Instead, look for direct children like 'data1', 'probe', etc.
                
                # Check data group
                if 'data1' in nirs:
                    print("\nDetails of data1 group:")
                    data1 = nirs['data1']
                    print(f"data1 keys: {list(data1.keys())}")
                    
                    # Examine dataTimeSeries if present
                    if 'dataTimeSeries' in data1:
                        print("\nShape of dataTimeSeries:")
                        dt = data1['dataTimeSeries']
                        print(dt.shape)
                        print(f"Data type: {dt.dtype}")
                        print("Sample values (first 5):", dt[:5])

                    # Check measurement list
                    if 'measurementList1' in data1:
                        print("\nMeasurement list information:")
                        ml = data1['measurementList1']
                        print(f"measurementList1 keys: {list(ml.keys())}")
                
                # Check probe information
                if 'probe' in nirs:
                    print("\nProbe information:")
                    probe = nirs['probe']
                    print(f"Probe keys: {list(probe.keys())}")
                    if 'sourcePos' in probe and 'detectorPos' in probe:
                        print(f"Number of sources: {len(probe['sourcePos'])}")
                        print(f"Number of detectors: {len(probe['detectorPos'])}")
    
    except Exception as e:
        print(f"Error inspecting file: {str(e)}")

if __name__ == "__main__":
    file_path = sys.argv[1] if len(sys.argv) > 1 else "/Users/clem/Desktop/code/data-pipeline/ref/snirfs/test_Test_6d1af09_MOMENTS.snirf"
    inspect_h5_structure(file_path)