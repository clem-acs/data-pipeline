#!/usr/bin/env python3
"""
Script to read metadata from SNIRF files.
This works with all SNIRF data types, including TD-fNIRS files.
Updated to handle flat structure and large measurement list counts.
"""
import os
import glob
import sys
import numpy as np
import h5py

def read_snirf_metadata(file_path):
    """Read and return metadata from a SNIRF file."""
    print(f"\nReading metadata from: {os.path.basename(file_path)}")
    try:
        # Open SNIRF file with h5py for direct access
        with h5py.File(file_path, 'r') as snirf_file:
            # Print basic information
            if 'formatVersion' in snirf_file:
                print(f"  Format version: {snirf_file['formatVersion'][()]}")
            
            # Check if nirs group exists
            if 'nirs' not in snirf_file:
                print("  Error: No 'nirs' group found in the file.")
                return None
            
            nirs = snirf_file['nirs']
            
            # Get all data elements (patterns like data1, data2, etc.)
            data_elements = [key for key in nirs.keys() if key.startswith('data')]
            print(f"  Number of data elements: {len(data_elements)}")
            
            # Process each data element
            for data_key in data_elements:
                data_element = nirs[data_key]
                print(f"\n  Data Element: {data_key}")
                
                # Data time series information
                if 'dataTimeSeries' in data_element:
                    dt = data_element['dataTimeSeries']
                    shape = dt.shape
                    print(f"    Data Shape: {shape}")
                    print(f"    Time Points: {shape[0]}")
                    print(f"    Channels: {shape[1]}")
                    
                    # Check for invalid values
                    has_nan = np.isnan(dt).any()
                    has_inf = np.isinf(dt).any()
                    print(f"    Contains NaN: {has_nan}")
                    print(f"    Contains Inf: {has_inf}")
                    
                    # Count valid channels (containing at least one finite, non-zero value)
                    valid_channels = 0
                    for col in range(min(shape[1], 1000)):  # Limit check to first 1000 channels for performance
                        channel_data = dt[:, col]
                        if np.any(np.isfinite(channel_data) & (channel_data != 0)):
                            valid_channels += 1
                    
                    valid_percent = (valid_channels / min(shape[1], 1000)) * 100
                    print(f"    Valid Channels (sample): {valid_channels}/{min(shape[1], 1000)} ({valid_percent:.2f}%)")
                
                # Measurement list information
                ml_keys = [key for key in data_element.keys() if key.startswith('measurementList')]
                print(f"    Total Measurement Lists: {len(ml_keys)}")
                
                if ml_keys:
                    # Sample the first few measurement lists
                    print("    Sample Measurements (first 5):")
                    for i, ml_key in enumerate(sorted(ml_keys)[:5]):
                        ml = data_element[ml_key]
                        
                        # Extract measurement properties
                        source_idx = ml['sourceIndex'][()] if 'sourceIndex' in ml else 'N/A'
                        detector_idx = ml['detectorIndex'][()] if 'detectorIndex' in ml else 'N/A'
                        data_type = ml['dataType'][()] if 'dataType' in ml else 'N/A'
                        data_type_name = get_data_type_name(data_type) if isinstance(data_type, (int, np.integer)) else 'Unknown'
                        wavelength = ml['wavelengthIndex'][()] if 'wavelengthIndex' in ml else 'N/A'
                        
                        print(f"      {ml_key}: Source: {source_idx}, Detector: {detector_idx}, "
                              f"Type: {data_type} ({data_type_name}), Wavelength: {wavelength}")
            
            # Probe information
            if 'probe' in nirs:
                probe = nirs['probe']
                print("\n  Probe Information:")
                
                # Source and detector positions
                if 'sourcePos3D' in probe:
                    sources = probe['sourcePos3D']
                    print(f"    Sources: {sources.shape[0]}")
                elif 'sourcePos' in probe:
                    sources = probe['sourcePos']
                    print(f"    Sources: {sources.shape[0] if hasattr(sources, 'shape') else 'Unknown'}")
                
                if 'detectorPos3D' in probe:
                    detectors = probe['detectorPos3D']
                    print(f"    Detectors: {detectors.shape[0]}")
                elif 'detectorPos' in probe:
                    detectors = probe['detectorPos']
                    print(f"    Detectors: {detectors.shape[0] if hasattr(detectors, 'shape') else 'Unknown'}")
                
                # Wavelengths
                if 'wavelengths' in probe:
                    wavelengths = probe['wavelengths']
                    print(f"    Wavelengths: {wavelengths[:]}")
            
            # Stimulus information
            stim_keys = [key for key in nirs.keys() if key.startswith('stim')]
            if stim_keys:
                print(f"\n  Stimulus Markers: {len(stim_keys)}")
                for stim_key in stim_keys[:5]:  # Show up to first 5
                    stim = nirs[stim_key]
                    if 'name' in stim:
                        name = stim['name'][()]
                        name_str = name.decode('utf-8') if isinstance(name, bytes) else str(name)
                        print(f"    - {stim_key}: {name_str}")
            
            # Metadata tags if available
            if 'metaDataTags' in nirs:
                meta = nirs['metaDataTags']
                print("\n  Metadata Tags:")
                meta_keys = list(meta.keys())[:10]  # Show up to first 10 keys
                for key in meta_keys:
                    try:
                        value = meta[key][()]
                        value_str = value.decode('utf-8') if isinstance(value, bytes) else str(value)
                        if len(value_str) < 50:  # Only print short values
                            print(f"    - {key}: {value_str}")
                    except Exception as e:
                        print(f"    - {key}: Error retrieving value - {str(e)}")
    
    except Exception as e:
        print(f"  Error reading file: {str(e)}")
        return None

def get_data_type_name(code):
    """Convert SNIRF data type code to a descriptive name."""
    if 1 <= code <= 100:
        return "Raw - Continuous Wave (CW)"
    elif 101 <= code <= 200:
        return "Raw - Frequency Domain (FD)"
    elif 201 <= code <= 300:
        return "Raw - Time Domain - Gated (TD Gated)"
    elif 301 <= code <= 400:
        return "Raw - Time Domain - Moments (TD Moments)"
    elif 401 <= code <= 500:
        return "Raw - Diffuse Correlation Spectroscopy (DCS)"
    elif code == 99999:
        return "Processed"
    else:
        return "Unknown"

def main():
    """Find and process SNIRF files provided as arguments or in current directory."""
    # Check if file paths were provided as command-line arguments
    if len(sys.argv) > 1:
        snirf_files = sys.argv[1:]
        print(f"Processing {len(snirf_files)} SNIRF files from command-line arguments...")
    else:
        # Otherwise, scan current directory
        print("Scanning for SNIRF files in current directory...")
        snirf_files = glob.glob("*.snirf")
        
        if not snirf_files:
            print("No SNIRF files found in the current directory.")
            return
        
        print(f"Found {len(snirf_files)} SNIRF files.")
    
    for file_path in snirf_files:
        read_snirf_metadata(file_path)

if __name__ == "__main__":
    main()