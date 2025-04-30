#!/usr/bin/env python3
import h5py
import numpy as np
import sys

def check_data_structure(file_path):
    """Examine the structure of a SNIRF file to check data format assumptions."""
    try:
        print(f"Examining SNIRF file: {file_path}")
        with h5py.File(file_path, 'r') as f:
            # Check general structure
            print("\n1. General structure:")
            print(f"Top-level groups: {list(f.keys())}")
            
            if 'nirs' in f:
                nirs = f['nirs']
                print(f"nirs keys: {list(nirs.keys())}")
                
                # Check data section
                print("\n2. Data format:")
                if 'data1' in nirs:
                    data1 = nirs['data1']
                    print(f"data1 keys: {list(data1.keys())}")
                    
                    # Check dataTimeSeries shape and content
                    if 'dataTimeSeries' in data1:
                        dt = data1['dataTimeSeries']
                        print(f"dataTimeSeries shape: {dt.shape}")
                        print(f"dataTimeSeries data type: {dt.dtype}")
                        print(f"Contains NaN: {np.isnan(dt).any()}")
                        print(f"Contains Inf: {np.isinf(dt).any()}")
                        
                        # Check sample values to see ranges
                        sample = dt[:5, :5]  # First 5 rows, first 5 columns
                        print(f"Sample values (first 5x5):\n{sample}")
                else:
                    print("No data1 section found")
                
                # Check measurement list structure
                print("\n3. Measurement List structure:")
                if 'data1' in nirs:
                    ml_keys = [k for k in data1.keys() if k.startswith('measurementList')]
                    print(f"Number of measurement lists: {len(ml_keys)}")
                    
                    if ml_keys:
                        # Check first few measurement lists
                        for i in range(1, min(6, len(ml_keys) + 1)):
                            ml_key = f'measurementList{i}'
                            if ml_key in data1:
                                ml = data1[ml_key]
                                print(f"\n{ml_key} keys: {list(ml.keys())}")
                                for attr in ml.keys():
                                    try:
                                        value = ml[attr][()]
                                        print(f"  {attr}: {value}")
                                    except Exception as e:
                                        print(f"  {attr}: Error retrieving value - {str(e)}")
                
                # Check probe structure
                print("\n4. Probe structure:")
                if 'probe' in nirs:
                    probe = nirs['probe']
                    print(f"probe keys: {list(probe.keys())}")
                    
                    if 'sourcePos' in probe:
                        src_pos = probe['sourcePos']
                        print(f"sourcePos shape: {src_pos.shape if hasattr(src_pos, 'shape') else 'N/A'}")
                    
                    if 'detectorPos' in probe:
                        det_pos = probe['detectorPos']
                        print(f"detectorPos shape: {det_pos.shape if hasattr(det_pos, 'shape') else 'N/A'}")
                        
                    if 'wavelengths' in probe:
                        wavelengths = probe['wavelengths']
                        print(f"wavelengths: {wavelengths[:]}")
                        
            else:
                print("No 'nirs' group found in the file")
                
    except Exception as e:
        print(f"Error examining file: {str(e)}")

if __name__ == "__main__":
    file_path = sys.argv[1] if len(sys.argv) > 1 else "/Users/clem/Desktop/code/data-pipeline/ref/snirfs/test_Test_6d1af09_MOMENTS.snirf"
    check_data_structure(file_path)