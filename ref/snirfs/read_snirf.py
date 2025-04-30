#!/usr/bin/env python3
"""
Script to read metadata from SNIRF files.
This works with all SNIRF data types, including TD-fNIRS files.
"""
import os
import glob
import sys
import numpy as np
from snirf import Snirf

def read_snirf_metadata(file_path):
    """Read and return metadata from a SNIRF file."""
    print(f"\nReading metadata from: {os.path.basename(file_path)}")
    try:
        # Open SNIRF file
        snirf_file = Snirf(file_path, 'r')
        
        # Print basic information
        print(f"  Format version: {snirf_file.formatVersion}")
        print(f"  Number of NIRS datasets: {len(snirf_file.nirs)}")
        
        # Process each NIRS dataset
        for i, nirs_data in enumerate(snirf_file.nirs):
            print(f"\n  NIRS Dataset #{i+1}:")
            
            # Probe information
            if hasattr(nirs_data, 'probe'):
                n_sources = len(nirs_data.probe.sourcePos) if hasattr(nirs_data.probe, 'sourcePos') else 0
                n_detectors = len(nirs_data.probe.detectorPos) if hasattr(nirs_data.probe, 'detectorPos') else 0
                print(f"    Sources: {n_sources}, Detectors: {n_detectors}")
            
            # Data information
            if hasattr(nirs_data, 'data') and len(nirs_data.data) > 0:
                for j, data_element in enumerate(nirs_data.data):
                    print(f"    Data Element #{j+1}:")
                    
                    # Data type
                    if hasattr(data_element, 'dataType'):
                        data_type_code = data_element.dataType
                        data_type_name = get_data_type_name(data_type_code)
                        print(f"      Data Type: {data_type_code} ({data_type_name})")
                    
                    # Time series info
                    if hasattr(data_element, 'dataTimeSeries') and data_element.dataTimeSeries is not None:
                        shape = data_element.dataTimeSeries.shape
                        print(f"      Data Shape: {shape}")
                        print(f"      Time Points: {shape[0]}")
                        print(f"      Channels: {shape[1]}")
                        
                        # Count valid channels (containing at least one finite, non-zero value)
                        data = data_element.dataTimeSeries
                        valid_channels = 0
                        for col in range(shape[1]):
                            channel_data = data[:, col]
                            if np.any(np.isfinite(channel_data) & (channel_data != 0)):
                                valid_channels += 1
                        
                        valid_percent = (valid_channels / shape[1]) * 100
                        print(f"      Valid Channels: {valid_channels}/{shape[1]} ({valid_percent:.2f}%)")
                    
                    # Measurement list sample
                    if hasattr(data_element, 'measurementList') and len(data_element.measurementList) > 0:
                        print(f"      Total Measurements: {len(data_element.measurementList)}")
                        print("      Sample Measurements (first 5):")
                        for k, measurement in enumerate(data_element.measurementList[:5]):
                            source_idx = getattr(measurement, 'sourceIndex', 'N/A')
                            detector_idx = getattr(measurement, 'detectorIndex', 'N/A')
                            data_type = getattr(measurement, 'dataType', 'N/A')
                            wavelength = getattr(measurement, 'wavelengthIndex', 'N/A')
                            print(f"        - Source: {source_idx}, Detector: {detector_idx}, Type: {data_type}, Wavelength: {wavelength}")
            
            # Stimulus information
            if hasattr(nirs_data, 'stim') and len(nirs_data.stim) > 0:
                print(f"    Stimulus Markers: {len(nirs_data.stim)}")
                for k, stim in enumerate(nirs_data.stim):
                    if hasattr(stim, 'name'):
                        print(f"      - Stimulus {k+1}: {stim.name}")
            
            # Metadata if available
            if hasattr(nirs_data, 'metaDataTags') and nirs_data.metaDataTags:
                print("    Metadata Tags:")
                try:
                    # Try to handle metaDataTags as a dict-like object
                    if hasattr(nirs_data.metaDataTags, 'items'):
                        for tag, value in nirs_data.metaDataTags.items():
                            if isinstance(value, str) and len(value) < 50:  # Only print short values
                                print(f"      - {tag}: {value}")
                    else:
                        print(f"      - Available but can't be displayed as key-value pairs")
                except Exception as e:
                    print(f"      - Error accessing metadata: {str(e)}")
        
        # Close the file when done
        snirf_file.close()
        
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