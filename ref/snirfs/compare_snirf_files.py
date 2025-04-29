#!/usr/bin/env python
import h5py
import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def extract_key_info(snirf_file):
    """Extract key information from a SNIRF file"""
    info = {}
    
    with h5py.File(snirf_file, 'r') as f:
        # Get format version
        info['format_version'] = f['formatVersion'][()]
        if isinstance(info['format_version'], bytes):
            info['format_version'] = info['format_version'].decode('utf-8')
        
        # Get metadata
        metadata = {}
        if 'nirs/metaDataTags' in f:
            for key in f['nirs/metaDataTags'].keys():
                value = f['nirs/metaDataTags'][key][()]
                if isinstance(value, bytes):
                    value = value.decode('utf-8')
                metadata[key] = value
        info['metadata'] = metadata
        
        # Get data shapes
        data_shapes = []
        i = 1
        while f'nirs/data{i}' in f:
            data_group = f[f'nirs/data{i}']
            for key in data_group.keys():
                if key.startswith('dataTimeSeries'):
                    shape = data_group[key].shape
                    dtype = data_group[key].dtype
                    data_shapes.append({
                        'data_element': i,
                        'name': key,
                        'shape': shape,
                        'dtype': dtype,
                        'size_mb': data_group[key].size * dtype.itemsize / (1024 * 1024)
                    })
            i += 1
        info['data_shapes'] = data_shapes
        
        # Get number of measurements
        measurements = []
        i = 1
        while f'nirs/data{i}' in f:
            data_group = f[f'nirs/data{i}']
            measurement_count = 0
            data_types = set()
            source_detector_pairs = set()
            
            for key in data_group.keys():
                if key.startswith('measurementList'):
                    measurement_count += 1
                    
                    # Extract data type
                    if 'dataType' in data_group[key]:
                        data_type = data_group[key]['dataType'][()]
                        data_types.add(data_type)
                    
                    # Extract source-detector pair
                    if 'sourceIndex' in data_group[key] and 'detectorIndex' in data_group[key]:
                        source = data_group[key]['sourceIndex'][()]
                        detector = data_group[key]['detectorIndex'][()]
                        source_detector_pairs.add((source, detector))
            
            measurements.append({
                'data_element': i,
                'count': measurement_count,
                'unique_data_types': list(data_types),
                'unique_source_detector_pairs': len(source_detector_pairs)
            })
            i += 1
        info['measurements'] = measurements
        
        # Get auxiliary data
        aux_data = []
        for key in f['nirs'].keys():
            if key.startswith('aux'):
                aux_group = f['nirs'][key]
                if 'name' in aux_group and 'dataTimeSeries' in aux_group:
                    name = aux_group['name'][()]
                    if isinstance(name, bytes):
                        name = name.decode('utf-8')
                    shape = aux_group['dataTimeSeries'].shape
                    aux_data.append({
                        'name': name,
                        'shape': shape
                    })
        info['aux_data'] = aux_data
        
        # Get probe information
        if 'nirs/probe' in f:
            probe = {}
            probe_group = f['nirs/probe']
            
            if 'wavelengths' in probe_group:
                probe['wavelengths'] = probe_group['wavelengths'][()]
            
            if 'sourceLabels' in probe_group:
                probe['source_count'] = len(probe_group['sourceLabels'])
            
            if 'detectorLabels' in probe_group:
                probe['detector_count'] = len(probe_group['detectorLabels'])
            
            info['probe'] = probe
    
    return info

def compare_snirf_files():
    """Compare all SNIRF files in the current directory"""
    snirf_files = list(Path('.').glob('*.snirf'))
    print(f"Found {len(snirf_files)} SNIRF files")
    
    results = {}
    
    for snirf_file in snirf_files:
        print(f"\nExtracting information from: {snirf_file}")
        try:
            info = extract_key_info(snirf_file)
            results[str(snirf_file)] = info
        except Exception as e:
            print(f"Error processing {snirf_file}: {e}")
    
    # Print comparison
    print("\n\n===== SNIRF FILES COMPARISON =====")
    
    # Compare format versions
    print("\nFormat Versions:")
    for file_name, info in results.items():
        print(f"  {file_name}: {info['format_version']}")
    
    # Compare data shapes
    print("\nData Shapes:")
    for file_name, info in results.items():
        print(f"  {file_name}:")
        for shape_info in info['data_shapes']:
            print(f"    Data Element #{shape_info['data_element']} - {shape_info['name']}: {shape_info['shape']} ({shape_info['size_mb']:.2f} MB)")
    
    # Compare measurement counts
    print("\nMeasurement Counts:")
    for file_name, info in results.items():
        print(f"  {file_name}:")
        for measurement in info['measurements']:
            print(f"    Data Element #{measurement['data_element']}: {measurement['count']} measurements")
            print(f"      Unique Data Types: {measurement['unique_data_types']}")
            print(f"      Unique Source-Detector Pairs: {measurement['unique_source_detector_pairs']}")
    
    # Compare aux data
    print("\nAuxiliary Data:")
    for file_name, info in results.items():
        print(f"  {file_name}:")
        for aux in info['aux_data'][:5]:  # Show only first 5 for brevity
            print(f"    {aux['name']}: {aux['shape']}")
        if len(info['aux_data']) > 5:
            print(f"    ... and {len(info['aux_data']) - 5} more")
    
    # Compare probe information
    print("\nProbe Information:")
    for file_name, info in results.items():
        if 'probe' in info:
            probe = info['probe']
            print(f"  {file_name}:")
            print(f"    Sources: {probe.get('source_count', 'N/A')}")
            print(f"    Detectors: {probe.get('detector_count', 'N/A')}")
            print(f"    Wavelengths: {probe.get('wavelengths', 'N/A')}")
    
    return results

if __name__ == "__main__":
    results = compare_snirf_files()
    
    # Plot data sizes for comparison
    file_names = list(results.keys())
    data_sizes = []
    
    for file_name in file_names:
        total_size = sum(shape_info['size_mb'] for shape_info in results[file_name]['data_shapes'])
        data_sizes.append(total_size)
    
    plt.figure(figsize=(10, 6))
    plt.bar(file_names, data_sizes)
    plt.title('SNIRF File Data Sizes (MB)')
    plt.xlabel('File')
    plt.ylabel('Size (MB)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('snirf_data_sizes.png')
    print("\nData size comparison chart saved as 'snirf_data_sizes.png'")