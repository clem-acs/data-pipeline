#!/usr/bin/env python3
"""
Utility functions for extracting data from SNIRF files, particularly focused on
TD-fNIRS MOMENTS data format.
"""
import h5py
import numpy as np
import os
from typing import Dict, Tuple, List, Optional, Any, Union

class SnirfDataExtractor:
    """Extracts and processes data from SNIRF files."""
    
    def __init__(self, file_path: str):
        """Initialize with SNIRF file path."""
        self.file_path = file_path
        self.file = None
        self._nirs = None
        self._data_elements = None
        self._format_version = None
        
    def __enter__(self):
        """Context manager enter method."""
        self.file = h5py.File(self.file_path, 'r')
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit method."""
        if self.file:
            self.file.close()
            
    @property
    def format_version(self) -> str:
        """Get the SNIRF format version."""
        if self._format_version is None and self.file:
            if 'formatVersion' in self.file:
                version = self.file['formatVersion'][()]
                self._format_version = version.decode('utf-8') if isinstance(version, bytes) else str(version)
            else:
                self._format_version = "Unknown"
        return self._format_version
    
    @property
    def nirs(self):
        """Get the NIRS group."""
        if self._nirs is None and self.file:
            if 'nirs' in self.file:
                self._nirs = self.file['nirs']
            else:
                raise ValueError("No 'nirs' group found in the SNIRF file")
        return self._nirs
    
    @property
    def data_elements(self) -> List[str]:
        """Get all data element keys (e.g., 'data1', 'data2', etc.)."""
        if self._data_elements is None and self.nirs:
            self._data_elements = [key for key in self.nirs.keys() if key.startswith('data')]
        return self._data_elements
    
    def get_data_time_series(self, data_key: str = 'data1') -> np.ndarray:
        """
        Get the time series data for a given data element.
        
        Args:
            data_key: The data element key (default: 'data1')
            
        Returns:
            NumPy array of shape (time_points, channels)
        """
        if not self.nirs or data_key not in self.nirs:
            raise ValueError(f"Data element '{data_key}' not found")
            
        data_element = self.nirs[data_key]
        if 'dataTimeSeries' not in data_element:
            raise ValueError(f"No dataTimeSeries found in {data_key}")
            
        return data_element['dataTimeSeries'][:]
    
    def get_measurement_info(self, data_key: str = 'data1') -> List[Dict[str, Any]]:
        """
        Get information about all measurements for a data element.
        
        Args:
            data_key: The data element key (default: 'data1')
            
        Returns:
            List of dictionaries with measurement information
        """
        if not self.nirs or data_key not in self.nirs:
            raise ValueError(f"Data element '{data_key}' not found")
            
        data_element = self.nirs[data_key]
        ml_keys = sorted([key for key in data_element.keys() if key.startswith('measurementList')])
        
        measurements = []
        for ml_key in ml_keys:
            ml = data_element[ml_key]
            measurement = {}
            
            # Extract common attributes
            for attr in ['sourceIndex', 'detectorIndex', 'dataType', 'dataTypeIndex', 'wavelengthIndex']:
                if attr in ml:
                    value = ml[attr][()]
                    measurement[attr] = value
                    
            # Handle string/byte attributes
            for attr in ['dataTypeLabel', 'dataUnit']:
                if attr in ml:
                    value = ml[attr][()]
                    measurement[attr] = value.decode('utf-8') if isinstance(value, bytes) else value
                    
            measurements.append(measurement)
            
        return measurements
    
    def get_probe_info(self) -> Dict[str, Any]:
        """
        Get information about the probe.
        
        Returns:
            Dictionary with probe information
        """
        if not self.nirs or 'probe' not in self.nirs:
            raise ValueError("No probe information found")
            
        probe = self.nirs['probe']
        probe_info = {}
        
        # Get wavelengths
        if 'wavelengths' in probe:
            probe_info['wavelengths'] = probe['wavelengths'][:]
            
        # Get source positions
        if 'sourcePos3D' in probe:
            probe_info['source_positions'] = probe['sourcePos3D'][:]
        elif 'sourcePos' in probe:
            probe_info['source_positions'] = probe['sourcePos'][:]
            
        # Get detector positions
        if 'detectorPos3D' in probe:
            probe_info['detector_positions'] = probe['detectorPos3D'][:]
        elif 'detectorPos' in probe:
            probe_info['detector_positions'] = probe['detectorPos'][:]
            
        # Get source and detector labels if available
        for key, new_key in [('sourceLabels', 'source_labels'), ('detectorLabels', 'detector_labels')]:
            if key in probe:
                # Handle both string arrays and individual string datasets
                labels = probe[key]
                if hasattr(labels, 'shape') and labels.shape:
                    probe_info[new_key] = [
                        label.decode('utf-8') if isinstance(label, bytes) else str(label)
                        for label in labels
                    ]
                else:
                    label = labels[()]
                    probe_info[new_key] = label.decode('utf-8') if isinstance(label, bytes) else str(label)
                    
        return probe_info
    
    def get_metadata(self) -> Dict[str, str]:
        """
        Get metadata from the SNIRF file.
        
        Returns:
            Dictionary with metadata tags
        """
        if not self.nirs or 'metaDataTags' not in self.nirs:
            return {}
            
        meta = self.nirs['metaDataTags']
        metadata = {}
        
        for key in meta.keys():
            try:
                value = meta[key][()]
                value_str = value.decode('utf-8') if isinstance(value, bytes) else str(value)
                metadata[key] = value_str
            except Exception:
                # Skip metadata that can't be retrieved
                pass
                
        return metadata
    
    def get_channel_mapping(self, data_key: str = 'data1') -> Dict[int, Dict[str, Any]]:
        """
        Create a mapping between channel indices and measurement properties.
        
        Args:
            data_key: The data element key (default: 'data1')
            
        Returns:
            Dictionary mapping channel indices to measurement information
        """
        measurements = self.get_measurement_info(data_key)
        channel_map = {}
        
        for i, measurement in enumerate(measurements):
            channel_map[i] = measurement
            
        return channel_map
    
    def extract_valid_data(self, data_key: str = 'data1', 
                          min_amplitude: float = 0.0) -> Tuple[np.ndarray, List[int]]:
        """
        Extract valid data channels that meet minimum quality criteria.
        
        Args:
            data_key: The data element key (default: 'data1')
            min_amplitude: Minimum amplitude threshold for a valid channel
            
        Returns:
            Tuple of (cleaned data array, list of valid channel indices)
        """
        data = self.get_data_time_series(data_key)
        
        # Find valid channels (containing at least one finite, non-zero value above threshold)
        valid_channels = []
        for col in range(data.shape[1]):
            channel_data = data[:, col]
            if np.any(np.isfinite(channel_data) & (channel_data > min_amplitude)):
                valid_channels.append(col)
                
        # Extract only the valid channels
        valid_data = data[:, valid_channels]
        
        return valid_data, valid_channels

def extract_data_from_snirf(file_path: str) -> Dict[str, Any]:
    """
    Convenience function to extract essential data from a SNIRF file.
    
    Args:
        file_path: Path to the SNIRF file
        
    Returns:
        Dictionary with extracted data and metadata
    """
    result = {
        'file_path': file_path,
        'file_name': os.path.basename(file_path),
    }
    
    with SnirfDataExtractor(file_path) as extractor:
        # Basic file info
        result['format_version'] = extractor.format_version
        
        # Metadata
        result['metadata'] = extractor.get_metadata()
        
        # Probe info
        try:
            result['probe_info'] = extractor.get_probe_info()
        except ValueError:
            result['probe_info'] = {}
            
        # Data info
        if extractor.data_elements:
            data_key = extractor.data_elements[0]  # Use the first data element
            
            # Get data shape
            try:
                data = extractor.get_data_time_series(data_key)
                result['data_shape'] = data.shape
                
                # Add data statistics
                result['data_stats'] = {
                    'has_nan': np.isnan(data).any(),
                    'has_inf': np.isinf(data).any(),
                    'min_value': float(np.nanmin(data)),
                    'max_value': float(np.nanmax(data)),
                    'mean_value': float(np.nanmean(data)),
                }
            except ValueError:
                result['data_shape'] = None
                result['data_stats'] = {}
                
            # Get sample measurements
            try:
                measurements = extractor.get_measurement_info(data_key)
                result['total_measurements'] = len(measurements)
                result['sample_measurements'] = measurements[:5] if measurements else []
            except ValueError:
                result['total_measurements'] = 0
                result['sample_measurements'] = []
                
    return result

if __name__ == "__main__":
    import sys
    import json
    
    if len(sys.argv) < 2:
        print("Usage: python snirf_data_extractor.py <path/to/snirf_file>")
        sys.exit(1)
        
    file_path = sys.argv[1]
    
    # Extract and display data summary
    data = extract_data_from_snirf(file_path)
    
    # Print basic summary
    print(f"File: {data['file_name']}")
    print(f"Format version: {data['format_version']}")
    if 'metadata' in data and data['metadata']:
        print("\nMetadata:")
        for key, value in data['metadata'].items():
            print(f"  {key}: {value}")
    
    if 'data_shape' in data and data['data_shape']:
        print(f"\nData shape: {data['data_shape']}")
        print(f"Total measurements: {data['total_measurements']}")
        
    if 'probe_info' in data and 'wavelengths' in data['probe_info']:
        wavelengths = data['probe_info']['wavelengths']
        if isinstance(wavelengths, np.ndarray):
            wavelengths = wavelengths.tolist()
        print(f"\nWavelengths: {wavelengths}")
    
    # Create simplified data structure for JSON
    json_data = {
        "file_info": {
            "file_name": data['file_name'],
            "file_path": data['file_path'],
            "format_version": data['format_version']
        },
        "metadata": {k: str(v) for k, v in data.get('metadata', {}).items()},
        "data_info": {
            "shape": list(data.get('data_shape', [])),
            "total_measurements": data.get('total_measurements', 0)
        },
        "data_stats": {}
    }
    
    # Add data statistics if available
    if 'data_stats' in data:
        stats = data['data_stats']
        json_data["data_stats"] = {
            "has_nan": bool(stats.get('has_nan', False)),
            "has_inf": bool(stats.get('has_inf', False)),
            "min_value": float(stats.get('min_value', 0)),
            "max_value": float(stats.get('max_value', 0)),
            "mean_value": float(stats.get('mean_value', 0))
        }
    
    # Add probe information
    if 'probe_info' in data and 'wavelengths' in data['probe_info']:
        wavelengths = data['probe_info']['wavelengths']
        if isinstance(wavelengths, np.ndarray):
            wavelengths = wavelengths.tolist()
        json_data["probe_info"] = {"wavelengths": wavelengths}
        
        # Add source and detector counts if available
        if 'source_positions' in data['probe_info']:
            source_pos = data['probe_info']['source_positions']
            json_data["probe_info"]["num_sources"] = len(source_pos) if isinstance(source_pos, list) else source_pos.shape[0]
            
        if 'detector_positions' in data['probe_info']:
            detector_pos = data['probe_info']['detector_positions']
            json_data["probe_info"]["num_detectors"] = len(detector_pos) if isinstance(detector_pos, list) else detector_pos.shape[0]
    
    # Add sample measurement info
    if 'sample_measurements' in data and data['sample_measurements']:
        json_data["sample_measurements"] = []
        for m in data['sample_measurements'][:5]:
            safe_m = {}
            for k, v in m.items():
                if isinstance(v, (np.integer, np.int64)):
                    safe_m[k] = int(v)
                elif isinstance(v, np.floating):
                    safe_m[k] = float(v)
                elif isinstance(v, np.bool_):
                    safe_m[k] = bool(v)
                elif isinstance(v, bytes):
                    safe_m[k] = v.decode('utf-8')
                else:
                    safe_m[k] = str(v)
            json_data["sample_measurements"].append(safe_m)
    
    # Write simplified JSON file
    output_file = os.path.splitext(file_path)[0] + "_summary.json"
    with open(output_file, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"\nDetailed summary written to: {output_file}")