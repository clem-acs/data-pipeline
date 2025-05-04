#!/usr/bin/env python3
"""
Script to filter fNIRS channels based on specific criteria and generate valid channel indices.
This allows for selecting a subset of channels based on source/detector positions,
wavelengths, moments, and other parameters.

The script can also generate valid channel indices using the same formula as in 
ref/s3h5fnirs/generate_valid_indices.py, allowing compatibility with existing processing pipelines.
"""
import os
import json
import numpy as np
import h5py
from pathlib import Path
import argparse
from scipy.spatial.distance import cdist

class FNIRSChannelFilter:
    """Filter fNIRS channels based on various criteria."""
    
    def __init__(self, snirf_file_path, layout_json_path=None):
        """
        Initialize with SNIRF file path and optional layout.json path.
        
        Args:
            snirf_file_path: Path to the SNIRF file
            layout_json_path: Optional path to layout.json for additional position metadata
        """
        self.snirf_file_path = snirf_file_path
        self.layout_json_path = layout_json_path
        self.snirf_file = None
        self.layout_data = None
        
        # Load layout.json if provided
        if layout_json_path:
            with open(layout_json_path, 'r') as f:
                self.layout_data = json.load(f)
        
        # Cached data
        self._measurements = None
        self._source_positions = None
        self._detector_positions = None
        self._source_group_mapping = None
        self._detector_group_mapping = None
        self._channel_mapping = None
        
    def __enter__(self):
        """Context manager enter method."""
        self.snirf_file = h5py.File(self.snirf_file_path, 'r')
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit method."""
        if self.snirf_file:
            self.snirf_file.close()
    
    def get_data_element_keys(self):
        """Get all data element keys (e.g., 'data1', 'data2', etc.)."""
        if not self.snirf_file or 'nirs' not in self.snirf_file:
            return []
        
        nirs = self.snirf_file['nirs']
        return [key for key in nirs.keys() if key.startswith('data')]
    
    def get_measurements(self, data_key='data1'):
        """
        Get all measurement information for a specific data element.
        
        Args:
            data_key: The data element key (default: 'data1')
            
        Returns:
            List of dictionaries with measurement information
        """
        if self._measurements is not None:
            return self._measurements
        
        if not self.snirf_file or 'nirs' not in self.snirf_file:
            return []
        
        nirs = self.snirf_file['nirs']
        if data_key not in nirs:
            return []
        
        data_element = nirs[data_key]
        ml_keys = sorted([key for key in data_element.keys() if key.startswith('measurementList')])
        
        measurements = []
        for i, ml_key in enumerate(ml_keys):
            ml = data_element[ml_key]
            measurement = {'channelIndex': i}  # Add channel index for easy reference
            
            # Extract common attributes
            for attr in ['sourceIndex', 'detectorIndex', 'dataType', 'dataTypeIndex', 'wavelengthIndex']:
                if attr in ml:
                    value = ml[attr][()]
                    measurement[attr] = int(value)
            
            # Handle string/byte attributes
            for attr in ['dataTypeLabel', 'dataUnit']:
                if attr in ml:
                    value = ml[attr][()]
                    measurement[attr] = value.decode('utf-8') if isinstance(value, bytes) else value
            
            measurements.append(measurement)
        
        self._measurements = measurements
        return measurements
    
    def get_source_positions(self):
        """Get source positions from the SNIRF file."""
        if self._source_positions is not None:
            return self._source_positions
        
        if not self.snirf_file or 'nirs' not in self.snirf_file:
            return []
        
        nirs = self.snirf_file['nirs']
        if 'probe' not in nirs:
            return []
        
        probe = nirs['probe']
        positions = []
        
        # Extract 3D positions
        if 'sourcePos3D' in probe:
            source_pos = probe['sourcePos3D'][:]
            for i, pos in enumerate(source_pos):
                positions.append({
                    'index': i + 1,  # 1-based indexing in SNIRF
                    'pos': [float(pos[0]), float(pos[1]), float(pos[2])],
                    'type': '3D'
                })
        # Fall back to 2D positions if 3D not available
        elif 'sourcePos' in probe:
            source_pos = probe['sourcePos'][:]
            for i, pos in enumerate(source_pos):
                positions.append({
                    'index': i + 1,
                    'pos': [float(pos[0]), float(pos[1]), float(pos[2]) if len(pos) > 2 else 0.0],
                    'type': '2D' if len(pos) <= 2 else '3D'
                })
        
        self._source_positions = positions
        return positions
    
    def get_detector_positions(self):
        """Get detector positions from the SNIRF file."""
        if self._detector_positions is not None:
            return self._detector_positions
        
        if not self.snirf_file or 'nirs' not in self.snirf_file:
            return []
        
        nirs = self.snirf_file['nirs']
        if 'probe' not in nirs:
            return []
        
        probe = nirs['probe']
        positions = []
        
        # Extract 3D positions
        if 'detectorPos3D' in probe:
            detector_pos = probe['detectorPos3D'][:]
            for i, pos in enumerate(detector_pos):
                positions.append({
                    'index': i + 1,  # 1-based indexing in SNIRF
                    'pos': [float(pos[0]), float(pos[1]), float(pos[2])],
                    'type': '3D'
                })
        # Fall back to 2D positions if 3D not available
        elif 'detectorPos' in probe:
            detector_pos = probe['detectorPos'][:]
            for i, pos in enumerate(detector_pos):
                positions.append({
                    'index': i + 1,
                    'pos': [float(pos[0]), float(pos[1]), float(pos[2]) if len(pos) > 2 else 0.0],
                    'type': '2D' if len(pos) <= 2 else '3D'
                })
        
        self._detector_positions = positions
        return positions
    
    def map_sources_to_groups(self):
        """
        Map SNIRF source indices to layout.json groups.
        
        Returns:
            Dictionary mapping SNIRF source indices to (group, position) tuples
        """
        if self._source_group_mapping is not None:
            return self._source_group_mapping
        
        if not self.layout_data:
            return {}
        
        source_positions = self.get_source_positions()
        if not source_positions:
            return {}
        
        # Create arrays for distance calculation
        snirf_pos_array = np.array([s['pos'] for s in source_positions])
        
        # Create layout position array and mapping
        layout_pos_array = []
        layout_mapping = []
        
        for group_idx, group in enumerate(self.layout_data.get("source_locations", [])):
            for pos_idx, pos in enumerate(group):
                layout_pos_array.append(pos)
                layout_mapping.append((group_idx + 1, pos_idx + 1))  # Store (group, position)
        
        layout_pos_array = np.array(layout_pos_array)
        
        # Calculate distances and find best matches
        source_group_mapping = {}
        
        # For each SNIRF source, find the closest layout position
        for i, source in enumerate(source_positions):
            distances = cdist([source['pos']], layout_pos_array)[0]
            min_idx = np.argmin(distances)
            min_distance = distances[min_idx]
            
            # Only map if distance is small (positions match closely)
            if min_distance < 0.1:  # Small threshold for exact matches
                source_group_mapping[source['index']] = layout_mapping[min_idx]
        
        self._source_group_mapping = source_group_mapping
        return source_group_mapping
    
    def map_detectors_to_groups(self):
        """
        Map SNIRF detector indices to layout.json groups.
        
        Returns:
            Dictionary mapping SNIRF detector indices to (group, position) tuples
        """
        if self._detector_group_mapping is not None:
            return self._detector_group_mapping
        
        if not self.layout_data:
            return {}
        
        detector_positions = self.get_detector_positions()
        if not detector_positions:
            return {}
        
        # Create arrays for distance calculation
        snirf_pos_array = np.array([d['pos'] for d in detector_positions])
        
        # Create layout position array and mapping
        layout_pos_array = []
        layout_mapping = []
        
        for group_idx, group in enumerate(self.layout_data.get("detector_locations", [])):
            for pos_idx, pos in enumerate(group):
                layout_pos_array.append(pos)
                layout_mapping.append((group_idx + 1, pos_idx + 1))  # Store (group, position)
        
        layout_pos_array = np.array(layout_pos_array)
        
        # Calculate distances and find best matches
        detector_group_mapping = {}
        
        # For each SNIRF detector, find the closest layout position
        for i, detector in enumerate(detector_positions):
            distances = cdist([detector['pos']], layout_pos_array)[0]
            min_idx = np.argmin(distances)
            min_distance = distances[min_idx]
            
            # Only map if distance is small (positions match closely)
            if min_distance < 0.1:  # Small threshold for exact matches
                detector_group_mapping[detector['index']] = layout_mapping[min_idx]
        
        self._detector_group_mapping = detector_group_mapping
        return detector_group_mapping
    
    def create_channel_mapping(self, data_key='data1'):
        """
        Create comprehensive channel mapping with all available information.
        
        Args:
            data_key: The data element key (default: 'data1')
            
        Returns:
            List of dictionaries with channel information
        """
        if self._channel_mapping is not None:
            return self._channel_mapping
        
        measurements = self.get_measurements(data_key)
        source_group_mapping = self.map_sources_to_groups()
        detector_group_mapping = self.map_detectors_to_groups()
        
        channel_mapping = []
        
        for measurement in measurements:
            channel_info = measurement.copy()  # Start with basic measurement info
            
            # Add source group info if available
            source_idx = measurement.get('sourceIndex')
            if source_idx in source_group_mapping:
                source_group, source_pos = source_group_mapping[source_idx]
                channel_info['sourceGroup'] = source_group
                channel_info['sourcePosition'] = source_pos
            
            # Add detector group info if available
            detector_idx = measurement.get('detectorIndex')
            if detector_idx in detector_group_mapping:
                detector_group, detector_pos = detector_group_mapping[detector_idx]
                channel_info['detectorGroup'] = detector_group
                channel_info['detectorPosition'] = detector_pos
            
            # Add wavelength info (using 1-based indexing)
            wavelength_idx = measurement.get('wavelengthIndex')
            if wavelength_idx is not None:
                if wavelength_idx == 1:
                    channel_info['wavelength'] = 690.0
                    channel_info['wavelengthName'] = 'Red'
                elif wavelength_idx == 2:
                    channel_info['wavelength'] = 905.0
                    channel_info['wavelengthName'] = 'IR'
            
            # Add moment info (specific to the dataset's convention)
            data_type_idx = measurement.get('dataTypeIndex')
            if data_type_idx is not None:
                # Map to common moment names based on the established naming convention
                if data_type_idx == 1:
                    channel_info['momentName'] = 'Zeroth'
                    channel_info['momentIdx'] = 0
                elif data_type_idx == 2:
                    channel_info['momentName'] = 'First'
                    channel_info['momentIdx'] = 1
                elif data_type_idx == 3:
                    channel_info['momentName'] = 'Second'
                    channel_info['momentIdx'] = 2
                else:
                    channel_info['momentName'] = f'Unknown-{data_type_idx}'
                    channel_info['momentIdx'] = data_type_idx - 1
            
            # Generate human-readable channel name
            source_idx = measurement.get('sourceIndex')
            detector_idx = measurement.get('detectorIndex')
            wavelength_idx = measurement.get('wavelengthIndex', 0)
            wavelength_name = channel_info.get('wavelengthName', '')
            moment_idx = channel_info.get('momentIdx', 0)
            moment_name = channel_info.get('momentName', '')
            
            # Get source and detector module info
            source_group = channel_info.get('sourceGroup')
            source_pos = channel_info.get('sourcePosition')
            detector_group = channel_info.get('detectorGroup')
            detector_pos = channel_info.get('detectorPosition')
            
            if all([source_idx, detector_idx, wavelength_idx, source_group, source_pos, detector_group, detector_pos]):
                # Follow the established naming convention
                # W{wavelength_idx}({wavelength_name})_M{moment_idx}({moment_name})_S{source_module}_{source_id}_D{detector_module}_{detector_id}
                # Adjust wavelength indexing to match convention (0-based)
                wave_idx = wavelength_idx - 1  # SNIRF uses 1-based, convention uses 0-based
                
                channel_info['channelName'] = (
                    f"W{wave_idx}({wavelength_name})_"
                    f"M{moment_idx}({moment_name})_"
                    f"S{source_group}_{source_pos}_"
                    f"D{detector_group}_{detector_pos}"
                )
            
            channel_mapping.append(channel_info)
        
        self._channel_mapping = channel_mapping
        return channel_mapping
    
    def filter_channels(self, data_key='data1', 
                        wavelength_indices=None, 
                        moment_indices=None,
                        source_groups=None, 
                        detector_groups=None,
                        source_positions=None, 
                        detector_positions=None,
                        distance_threshold=None):
        """
        Filter channels based on multiple criteria.
        
        Args:
            data_key: The data element key (default: 'data1')
            wavelength_indices: List of wavelength indices to include (1-based, e.g., [1, 2])
            moment_indices: List of moment indices to include (1-based, e.g., [1, 2, 3])
            source_groups: List of source groups to include
            detector_groups: List of detector groups to include
            source_positions: List of source positions within groups to include
            detector_positions: List of detector positions within groups to include
            distance_threshold: Maximum distance between source and detector (in position units)
            
        Returns:
            List of filtered channel indices
        """
        channel_mapping = self.create_channel_mapping(data_key)
        filtered_indices = []
        
        for channel in channel_mapping:
            # Check wavelength filter
            if wavelength_indices and channel.get('wavelengthIndex') not in wavelength_indices:
                continue
            
            # Check moment filter (dataTypeIndex corresponds to moment)
            if moment_indices and channel.get('dataTypeIndex') not in moment_indices:
                continue
            
            # Check source group filter
            if source_groups and channel.get('sourceGroup') not in source_groups:
                continue
            
            # Check detector group filter
            if detector_groups and channel.get('detectorGroup') not in detector_groups:
                continue
            
            # Check source position filter
            if source_positions and channel.get('sourcePosition') not in source_positions:
                continue
            
            # Check detector position filter
            if detector_positions and channel.get('detectorPosition') not in detector_positions:
                continue
            
            # Check distance threshold if applicable
            if distance_threshold is not None:
                # Would require calculating distance between source and detector
                # Skip for now - would need additional implementation
                pass
            
            # Channel passed all filters
            filtered_indices.append(channel.get('channelIndex'))
        
        return filtered_indices
    
    def get_data_for_channels(self, filtered_indices, data_key='data1'):
        """
        Extract data for filtered channels.
        
        Args:
            filtered_indices: List of channel indices to include
            data_key: The data element key (default: 'data1')
            
        Returns:
            NumPy array of shape (time_points, channels)
        """
        if not self.snirf_file or 'nirs' not in self.snirf_file:
            return None
        
        nirs = self.snirf_file['nirs']
        if data_key not in nirs:
            return None
            
        data_element = nirs[data_key]
        if 'dataTimeSeries' not in data_element:
            return None
            
        # Get full data array
        full_data = data_element['dataTimeSeries'][:]
        
        # Extract only filtered channels
        if filtered_indices:
            return full_data[:, filtered_indices]
        else:
            return full_data
    
    def get_channel_info(self, filtered_indices):
        """
        Get detailed information for filtered channels.
        
        Args:
            filtered_indices: List of channel indices to include
            
        Returns:
            List of channel information dictionaries
        """
        channel_mapping = self.create_channel_mapping()
        return [channel_mapping[i] for i in filtered_indices if i < len(channel_mapping)]
    
    def save_filtered_channels(self, output_path, filtered_indices, data_key='data1'):
        """
        Save filtered channel data and information to JSON file.
        
        Args:
            output_path: Path to save the output JSON file
            filtered_indices: List of channel indices to include
            data_key: The data element key (default: 'data1')
        """
        channel_info = self.get_channel_info(filtered_indices)
        
        # Convert NumPy arrays and other non-JSON serializable objects to lists/strings
        serializable_info = []
        for channel in channel_info:
            clean_channel = {}
            for key, value in channel.items():
                if isinstance(value, np.ndarray):
                    clean_channel[key] = value.tolist()
                elif isinstance(value, np.integer):
                    clean_channel[key] = int(value)
                elif isinstance(value, np.floating):
                    clean_channel[key] = float(value)
                else:
                    clean_channel[key] = value
            serializable_info.append(clean_channel)
        
        # Save to JSON
        with open(output_path, 'w') as f:
            json.dump({
                'source_file': os.path.basename(self.snirf_file_path),
                'data_key': data_key,
                'total_channels': len(filtered_indices),
                'channels': serializable_info
            }, f, indent=2)
    
    def generate_filter_report(self, filtered_indices, file_path, 
                              wavelength_indices=None, 
                              moment_indices=None,
                              source_groups=None, 
                              detector_groups=None,
                              source_positions=None, 
                              detector_positions=None):
        """
        Generate a readable report of the filtered channels.
        
        Args:
            filtered_indices: List of filtered channel indices
            file_path: Path to save the report
            wavelength_indices, moment_indices, etc.: Filter criteria used
            
        Returns:
            Path to the generated report file
        """
        channel_info = self.get_channel_info(filtered_indices)
        
        report = f"# Filtered fNIRS Channel Report\n\n"
        report += f"**Source file:** {os.path.basename(self.snirf_file_path)}\n"
        report += f"**Total channels after filtering:** {len(filtered_indices)}\n\n"
        
        # Filter criteria summary
        report += "## Filter Criteria\n\n"
        if wavelength_indices:
            wavelength_names = {1: "690nm (Red)", 2: "905nm (IR)"}
            wavelength_str = ", ".join(
                [f"{i} ({wavelength_names.get(i, 'Unknown')})" for i in wavelength_indices]
            )
            report += f"- **Wavelengths:** {wavelength_str}\n"
        
        if moment_indices:
            moment_names = {1: "Zeroth", 2: "First", 3: "Second"}
            moment_str = ", ".join(
                [f"{i} ({moment_names.get(i, 'Unknown')})" for i in moment_indices]
            )
            report += f"- **Moments:** {moment_str}\n"
        
        if source_groups:
            report += f"- **Source Groups:** {', '.join(map(str, source_groups))}\n"
        
        if detector_groups:
            report += f"- **Detector Groups:** {', '.join(map(str, detector_groups))}\n"
        
        if source_positions:
            report += f"- **Source Positions:** {', '.join(map(str, source_positions))}\n"
        
        if detector_positions:
            report += f"- **Detector Positions:** {', '.join(map(str, detector_positions))}\n"
        
        # Group statistics
        report += "\n## Channel Distribution\n\n"
        
        # Count wavelengths
        wavelength_counts = {}
        for channel in channel_info:
            w_idx = channel.get('wavelengthIndex')
            if w_idx is not None:
                wavelength_counts[w_idx] = wavelength_counts.get(w_idx, 0) + 1
        
        report += "### Wavelength Distribution\n\n"
        report += "| Wavelength Index | Count | Percentage |\n"
        report += "|------------------|-------|------------|\n"
        for w_idx in sorted(wavelength_counts.keys()):
            count = wavelength_counts[w_idx]
            percentage = (count / len(channel_info)) * 100
            report += f"| {w_idx} | {count} | {percentage:.1f}% |\n"
        
        # Count moments
        moment_counts = {}
        for channel in channel_info:
            m_idx = channel.get('dataTypeIndex')
            if m_idx is not None:
                moment_counts[m_idx] = moment_counts.get(m_idx, 0) + 1
        
        report += "\n### Moment Distribution\n\n"
        report += "| Moment Index | Count | Percentage |\n"
        report += "|--------------|-------|------------|\n"
        for m_idx in sorted(moment_counts.keys()):
            count = moment_counts[m_idx]
            percentage = (count / len(channel_info)) * 100
            report += f"| {m_idx} | {count} | {percentage:.1f}% |\n"
        
        # Count source-detector pairs
        sd_pairs = set()
        for channel in channel_info:
            s_idx = channel.get('sourceIndex')
            d_idx = channel.get('detectorIndex')
            if s_idx is not None and d_idx is not None:
                sd_pairs.add((s_idx, d_idx))
        
        report += f"\n### Source-Detector Pairs\n\n"
        report += f"Total unique source-detector pairs: {len(sd_pairs)}\n\n"
        
        # Channel listing (first 20 for brevity)
        report += "\n## Channel Listing (First 20)\n\n"
        report += "| Index | Channel Name | Source | Detector | Wavelength | Moment |\n"
        report += "|-------|--------------|--------|----------|------------|--------|\n"
        
        for i, channel in enumerate(channel_info[:20]):
            channel_name = channel.get('channelName', 'Unknown')
            source = f"{channel.get('sourceIndex', 'N/A')} (Group {channel.get('sourceGroup', 'N/A')})"
            detector = f"{channel.get('detectorIndex', 'N/A')} (Group {channel.get('detectorGroup', 'N/A')})"
            wavelength = f"{channel.get('wavelength', 'N/A')}nm ({channel.get('wavelengthName', 'N/A')})"
            moment = channel.get('momentName', 'N/A')
            
            report += f"| {filtered_indices[i]} | {channel_name} | {source} | {detector} | {wavelength} | {moment} |\n"
        
        if len(channel_info) > 20:
            report += f"*...and {len(channel_info) - 20} more channels...*\n"
        
        # Write report to file
        with open(file_path, 'w') as f:
            f.write(report)
        
        return file_path


def calculate_channel_index(wavelength_idx, moment_idx, source_module, source_id, detector_module, detector_id):
    """
    Calculate the index for a specific channel based on the formula from generate_valid_indices.py:
    
    index = ((((wavelength_idx * 3 + moment_idx) * 48 + (source_module-1)) * 3 + (source_id-1)) * 48 + (detector_module-1)) * 6 + (detector_id-1)
    
    Parameters:
    - wavelength_idx: 0 (Red) or 1 (IR)
    - moment_idx: 0 (Zeroth), 1 (First), or 2 (Second)
    - source_module: 1-48
    - source_id: 1-3
    - detector_module: 1-48
    - detector_id: 1-6
    
    Returns:
    - index: The calculated channel index
    """
    index = ((((wavelength_idx * 3 + moment_idx) * 48 + (source_module-1)) * 3 + (source_id-1)) * 48 + (detector_module-1)) * 6 + (detector_id-1)
    return index


def generate_valid_indices(present_modules):
    """
    Generate all valid channel indices for source-detector combinations where
    both source and detector modules are present in the input list.
    
    Parameters:
    - present_modules: List of module numbers that are present (1-48)
    
    Returns:
    - indices: List of valid channel indices
    """
    indices = []
    
    # Iterate through all wavelengths (0=Red, 1=IR)
    for wavelength_idx in range(2):
        # Iterate through all moments (0=Zeroth, 1=First, 2=Second)
        for moment_idx in range(3):
            # Iterate through all present source modules
            for source_module in present_modules:
                # Iterate through all source IDs in the module (1-3)
                for source_id in range(1, 4):
                    # Iterate through all present detector modules
                    for detector_module in present_modules:
                        # Iterate through all detector IDs in the module (1-6)
                        for detector_id in range(1, 7):
                            # Calculate the index for this channel
                            index = calculate_channel_index(
                                wavelength_idx, moment_idx,
                                source_module, source_id,
                                detector_module, detector_id
                            )
                            indices.append(index)
    
    return sorted(indices)


def indices_by_category(present_modules):
    """
    Generate organized indices by wavelength and moment categories.
    
    Parameters:
    - present_modules: List of module numbers that are present (1-48)
    
    Returns:
    - categorized_indices: Dictionary with wavelength and moment categories
    """
    categorized_indices = {
        "wavelengths": {
            "Red": [],   # wavelength_idx = 0
            "IR": []     # wavelength_idx = 1
        },
        "moments": {
            "Zeroth": [],  # moment_idx = 0
            "First": [],   # moment_idx = 1
            "Second": []   # moment_idx = 2
        },
        "wavelength_moment_combinations": {
            "Red_Zeroth": [],
            "Red_First": [],
            "Red_Second": [],
            "IR_Zeroth": [],
            "IR_First": [],
            "IR_Second": []
        }
    }
    
    wavelength_names = ["Red", "IR"]
    moment_names = ["Zeroth", "First", "Second"]
    
    for wavelength_idx, wavelength_name in enumerate(wavelength_names):
        for moment_idx, moment_name in enumerate(moment_names):
            for source_module in present_modules:
                for source_id in range(1, 4):
                    for detector_module in present_modules:
                        for detector_id in range(1, 7):
                            index = calculate_channel_index(
                                wavelength_idx, moment_idx,
                                source_module, source_id,
                                detector_module, detector_id
                            )
                            
                            # Add to wavelength category
                            categorized_indices["wavelengths"][wavelength_name].append(index)
                            
                            # Add to moment category
                            categorized_indices["moments"][moment_name].append(index)
                            
                            # Add to combination category
                            combo_key = f"{wavelength_name}_{moment_name}"
                            categorized_indices["wavelength_moment_combinations"][combo_key].append(index)
    
    # Sort all lists of indices
    for category in categorized_indices:
        for key in categorized_indices[category]:
            categorized_indices[category][key].sort()
    
    return categorized_indices


def generate_channel_names(present_modules):
    """
    Generate the channel names for all valid source-detector combinations.
    
    Parameters:
    - present_modules: List of module numbers that are present (1-48)
    
    Returns:
    - channel_names: List of channel names
    """
    wavelength_names = ["Red", "IR"]
    moment_names = ["Zeroth", "First", "Second"]
    channel_names = []
    
    for wavelength_idx, wavelength_name in enumerate(wavelength_names):
        for moment_idx, moment_name in enumerate(moment_names):
            for source_module in present_modules:
                for source_id in range(1, 4):
                    for detector_module in present_modules:
                        for detector_id in range(1, 7):
                            # Format: W{wavelength_idx}({wavelength_name})_M{moment_idx}({moment_name})_S{source_module}_{source_id}_D{detector_module}_{detector_id}
                            channel_name = f"W{wavelength_idx}({wavelength_name})_M{moment_idx}({moment_name})_S{source_module}_{source_id}_D{detector_module}_{detector_id}"
                            channel_names.append((channel_name, calculate_channel_index(
                                wavelength_idx, moment_idx, source_module, source_id, detector_module, detector_id
                            )))
    
    # Sort by index
    channel_names.sort(key=lambda x: x[1])
    return [name for name, _ in channel_names]


def main():
    """Command-line interface for filtering fNIRS channels."""
    parser = argparse.ArgumentParser(description='Filter fNIRS channels based on criteria and generate valid indices')
    
    # Original SNIRF filtering options
    snirf_group = parser.add_argument_group('SNIRF Filtering Options')
    snirf_group.add_argument('--snirf', help='Path to SNIRF file')
    snirf_group.add_argument('--layout', default=None, help='Path to layout.json file')
    snirf_group.add_argument('--output', help='Output JSON file for filtered channels')
    snirf_group.add_argument('--report', help='Output MD file for filter report')
    snirf_group.add_argument('--wavelengths', type=int, nargs='+', 
                       help='Wavelength indices to include (1-based)')
    snirf_group.add_argument('--moments', type=int, nargs='+',
                       help='Moment indices to include (1-based)')
    snirf_group.add_argument('--source-groups', type=int, nargs='+',
                       help='Source groups to include')
    snirf_group.add_argument('--detector-groups', type=int, nargs='+',
                       help='Detector groups to include')
    snirf_group.add_argument('--source-positions', type=int, nargs='+',
                       help='Source positions within groups to include')
    snirf_group.add_argument('--detector-positions', type=int, nargs='+',
                       help='Detector positions within groups to include')
    snirf_group.add_argument('--data-key', default='data1',
                       help='Data element key in SNIRF file')
    
    # Index generation options from generate_valid_indices.py
    indices_group = parser.add_argument_group('Index Generation Options')
    indices_group.add_argument('--generate-indices', action='store_true',
                       help='Generate valid indices instead of filtering a SNIRF file')
    indices_group.add_argument('--modules', type=int, nargs='+',
                       help='List of present module numbers (1-48) for index generation')
    indices_group.add_argument('--generate-names', action='store_true',
                       help='Generate channel names instead of indices')
    indices_group.add_argument('--categorize', action='store_true',
                       help='Categorize indices by wavelength and moment')
    
    args = parser.parse_args()
    
    # Handle index generation mode
    if args.generate_indices:
        if not args.modules:
            print("Error: --modules argument is required with --generate-indices")
            return
            
        # Validate module numbers
        present_modules = []
        for module in args.modules:
            if module < 1 or module > 48:
                print(f"Warning: Module number {module} is out of range (1-48). Skipping.")
            else:
                present_modules.append(module)
        
        if not present_modules:
            print("Error: No valid module numbers provided.")
            return
            
        if args.categorize:
            # Generate categorized indices
            result = indices_by_category(present_modules)
            print(f"Generated categorized indices for modules: {present_modules}")
            
            if args.output:
                import json
                with open(args.output, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"Saved categorized indices to {args.output}")
            else:
                # Print a summary
                for category, subcategories in result.items():
                    print(f"\n{category.capitalize()}:")
                    for subcategory, indices in subcategories.items():
                        print(f"  - {subcategory}: {len(indices)} indices")
                        if len(indices) <= 5:
                            print(f"    {indices}")
                        else:
                            print(f"    {indices[:5]} ... (and {len(indices)-5} more)")
        
        elif args.generate_names:
            # Generate channel names
            channel_names = generate_channel_names(present_modules)
            print(f"Generated {len(channel_names)} channel names for modules: {present_modules}")
            
            if args.output:
                with open(args.output, 'w') as f:
                    for name in channel_names:
                        f.write(name + '\n')
                print(f"Saved channel names to {args.output}")
            else:
                # Print the first few channel names
                for i, name in enumerate(channel_names[:10]):
                    print(name)
                if len(channel_names) > 10:
                    print(f"... and {len(channel_names)-10} more channel names")
        
        else:
            # Generate plain list of indices
            indices = generate_valid_indices(present_modules)
            print(f"Generated {len(indices)} indices for modules: {present_modules}")
            
            if args.output:
                # Save indices to the specified output file
                np.save(args.output, np.array(indices))
                print(f"Saved indices to {args.output}")
            else:
                # Print the first few indices
                if len(indices) <= 20:
                    print(indices)
                else:
                    print(indices[:20])
                    print(f"... and {len(indices)-20} more indices")
    
    # Handle SNIRF filtering mode
    else:
        if not args.snirf:
            print("Error: --snirf argument is required when not using --generate-indices")
            return
            
        # Set default output paths if not provided
        if not args.output:
            output_dir = os.path.dirname(args.snirf)
            base_name = os.path.splitext(os.path.basename(args.snirf))[0]
            args.output = os.path.join(output_dir, f"{base_name}_filtered_channels.json")
        
        if not args.report:
            output_dir = os.path.dirname(args.snirf)
            base_name = os.path.splitext(os.path.basename(args.snirf))[0]
            args.report = os.path.join(output_dir, f"{base_name}_filtered_channels.md")
        
        # Apply filters
        with FNIRSChannelFilter(args.snirf, args.layout) as filter_obj:
            filtered_indices = filter_obj.filter_channels(
                data_key=args.data_key,
                wavelength_indices=args.wavelengths,
                moment_indices=args.moments,
                source_groups=args.source_groups,
                detector_groups=args.detector_groups,
                source_positions=args.source_positions,
                detector_positions=args.detector_positions
            )
            
            # Save results
            if filtered_indices:
                filter_obj.save_filtered_channels(args.output, filtered_indices, args.data_key)
                filter_obj.generate_filter_report(
                    filtered_indices, args.report,
                    wavelength_indices=args.wavelengths,
                    moment_indices=args.moments,
                    source_groups=args.source_groups,
                    detector_groups=args.detector_groups,
                    source_positions=args.source_positions,
                    detector_positions=args.detector_positions
                )
                
                print(f"Filtered {len(filtered_indices)} channels out of {len(filter_obj.get_measurements(args.data_key))}")
                print(f"Results saved to: {args.output}")
                print(f"Report saved to: {args.report}")
            else:
                print("No channels matched the filter criteria")


if __name__ == "__main__":
    main()