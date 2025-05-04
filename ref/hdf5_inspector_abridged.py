#!/usr/bin/env python3
"""
HDF5 Inspector Abridged - A more intelligent script to summarize HDF5 file structure
"""

import h5py
import numpy as np
import argparse
import os
import re
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any, Optional, Union

def summarize_attributes(item: Union[h5py.Group, h5py.Dataset]) -> str:
    """
    Summarize attributes of an HDF5 item (group or dataset)
    
    Args:
        item: The HDF5 item to summarize
        
    Returns:
        A string with key attribute information
    """
    if len(item.attrs) == 0:
        return "No attributes"
    
    # Count attribute types
    attr_types = Counter()
    for key, value in item.attrs.items():
        attr_types[type(value).__name__] += 1
    
    # Sample a few attributes (prioritize common types for neuro data)
    priority_attrs = ["device_type", "sequence_num", "timestamp", "device_timestamp", 
                      "server_timestamp", "block_size", "sfreq", "channel_names"]
    
    samples = []
    for key in priority_attrs:
        if key in item.attrs:
            samples.append(f"{key}: {item.attrs[key]}")
    
    # Add a few more if we haven't reached 3 samples yet
    if len(samples) < 3:
        for key, value in item.attrs.items():
            if key not in priority_attrs and len(samples) < 3:
                if isinstance(value, (np.ndarray, list)) and len(value) > 3:
                    samples.append(f"{key}: array[{len(value)}]")
                else:
                    samples.append(f"{key}: {value}")
    
    return f"{len(item.attrs)} attrs ({', '.join(f'{count} {type_}' for type_, count in attr_types.items())}), " + \
           f"e.g. {', '.join(samples[:3])}" + (f" + {len(item.attrs) - 3} more" if len(item.attrs) > 3 else "")

def get_dataset_stats(dataset: h5py.Dataset) -> Dict[str, Any]:
    """
    Get statistical information about a dataset
    
    Args:
        dataset: The HDF5 dataset
        
    Returns:
        A dictionary with shape, dtype, and basic statistical information
    """
    stats = {
        "shape": dataset.shape,
        "dtype": dataset.dtype,
        "size": dataset.size,
        "ndim": len(dataset.shape),
    }
    
    # For numerical data, calculate basic statistics (if dataset is not too large)
    if np.issubdtype(dataset.dtype, np.number) and dataset.size < 1e7:
        try:
            # Get a sample to calculate statistics
            if dataset.size > 1e5:
                # For large datasets, sample random elements
                if len(dataset.shape) == 1:
                    indices = np.random.choice(dataset.shape[0], size=10000, replace=False)
                    data_sample = dataset[indices]
                elif len(dataset.shape) == 2:
                    row_indices = np.random.choice(dataset.shape[0], size=100, replace=False)
                    col_indices = np.random.choice(dataset.shape[1], size=100, replace=False)
                    data_sample = dataset[row_indices][:, col_indices]
                else:
                    # For higher dimensions, just get first elements
                    slices = tuple(slice(0, min(10, dim)) for dim in dataset.shape)
                    data_sample = dataset[slices]
            else:
                data_sample = dataset[:]
            
            stats["min"] = np.min(data_sample)
            stats["max"] = np.max(data_sample)
            stats["mean"] = np.mean(data_sample)
            stats["std"] = np.std(data_sample)
            stats["has_zeros"] = (data_sample == 0).any()
            stats["has_nans"] = np.isnan(data_sample).any() if hasattr(data_sample, 'dtype') and np.issubdtype(data_sample.dtype, np.floating) else False
        except Exception as e:
            stats["error"] = str(e)
    
    return stats

def explore_structure(h5file: h5py.File) -> Dict[str, Any]:
    """
    Explore the structure of an HDF5 file and gather information
    
    Args:
        h5file: The HDF5 file to explore
        
    Returns:
        A dictionary with structured information about the file
    """
    structure = {
        "groups": {},
        "dataset_count": 0,
        "group_count": 0,
        "max_depth": 0,
        "large_datasets": [],
        "metadata_groups": [],
        "time_series_candidates": [],
        "group_name_patterns": [],
    }
    
    # Timestamp patterns for metadata groups
    timestamp_pattern = re.compile(r'\d{13,}')
    
    def explore_group(name, obj):
        if isinstance(obj, h5py.Group):
            # Count the group
            structure["group_count"] += 1
            
            # Track depth
            depth = name.count('/')
            structure["max_depth"] = max(structure["max_depth"], depth + 1)
            
            # Check for timestamp pattern in the name
            basename = name.split('/')[-1] if name else ''
            if timestamp_pattern.match(basename):
                structure["metadata_groups"].append(name)
            
            # Add to group hierarchy
            parent_path = '/'.join(name.split('/')[:-1]) if name else ''
            if parent_path not in structure["groups"]:
                structure["groups"][parent_path] = {"subgroups": [], "datasets": []}
            if name not in structure["groups"]:
                structure["groups"][name] = {"subgroups": [], "datasets": []}
            if name:
                structure["groups"][parent_path]["subgroups"].append(name)
        
        elif isinstance(obj, h5py.Dataset):
            # Count the dataset
            structure["dataset_count"] += 1
            
            # Track depth
            depth = name.count('/')
            structure["max_depth"] = max(structure["max_depth"], depth + 1)
            
            # Get parent group
            parent_path = '/'.join(name.split('/')[:-1]) if name else ''
            if parent_path not in structure["groups"]:
                structure["groups"][parent_path] = {"subgroups": [], "datasets": []}
            structure["groups"][parent_path]["datasets"].append(name)
            
            # Track large datasets
            if obj.size > 1e5:
                structure["large_datasets"].append((name, obj.shape, obj.dtype, obj.size))
            
            # Identify potential time series data
            if len(obj.shape) >= 2:
                # Look for a shape where one dimension is much larger than others
                max_dim = max(obj.shape)
                if max_dim > 1000 and max_dim / sum(obj.shape) > 0.8:
                    structure["time_series_candidates"].append((name, obj.shape, obj.dtype))
    
    # Visit all objects in the file
    h5file.visititems(explore_group)
    
    # Analyze group name patterns
    # Look for common prefixes and numeric suffixes
    group_names = []
    for path in structure["groups"].keys():
        if path:  # Skip root group
            basename = path.split('/')[-1]
            group_names.append(basename)
    
    # Find numeric patterns (like "frame_123")
    numeric_pattern = re.compile(r'(.+?)(\d+)$')
    pattern_groups = defaultdict(list)
    
    for name in group_names:
        match = numeric_pattern.match(name)
        if match:
            prefix, num = match.groups()
            pattern_groups[prefix].append(int(num))
    
    # Keep patterns with multiple occurrences
    for prefix, nums in pattern_groups.items():
        if len(nums) > 5:  # Only report patterns with sufficient occurrences
            structure["group_name_patterns"].append((prefix, len(nums), min(nums), max(nums)))
    
    return structure

def analyze_file(file_path: str, options: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze an HDF5 file and return structured information
    
    Args:
        file_path: Path to the HDF5 file
        options: Dictionary of analysis options
        
    Returns:
        A dictionary with analysis results
    """
    results = {
        "file_path": file_path,
        "file_size": os.path.getsize(file_path),
        "structure": None,
        "examined_datasets": [],
        "examined_groups": [],
    }
    
    try:
        with h5py.File(file_path, 'r') as f:
            # Explore overall structure
            results["structure"] = explore_structure(f)
            
            # Examine specific datasets if requested
            if options["examine_large_datasets"]:
                # Sort large datasets by size
                large_datasets = sorted(results["structure"]["large_datasets"], 
                                       key=lambda x: x[3], reverse=True)
                
                # Examine top N large datasets
                for name, shape, dtype, size in large_datasets[:options["max_datasets"]]:
                    dataset = f[name]
                    dataset_info = {
                        "name": name,
                        "stats": get_dataset_stats(dataset),
                        "attrs": summarize_attributes(dataset),
                    }
                    results["examined_datasets"].append(dataset_info)
            
            # Examine specific groups if requested
            if options["examine_metadata"]:
                # Get a sample of metadata groups
                metadata_groups = results["structure"]["metadata_groups"]
                if metadata_groups:
                    # Sort by name and sample evenly from the range
                    metadata_groups.sort()
                    step = max(1, len(metadata_groups) // min(10, len(metadata_groups)))
                    samples = metadata_groups[::step]
                    
                    for name in samples[:options["max_groups"]]:
                        group = f[name]
                        group_info = {
                            "name": name,
                            "attrs": summarize_attributes(group),
                            "contents": [(k, type(v).__name__) for k, v in group.items()]
                        }
                        results["examined_groups"].append(group_info)
    
    except Exception as e:
        results["error"] = str(e)
    
    return results

def print_file_summary(results: Dict[str, Any]) -> None:
    """
    Print a summary of the HDF5 file analysis
    
    Args:
        results: The analysis results dictionary
    """
    structure = results["structure"]
    
    print(f"\n=== HDF5 FILE SUMMARY ===")
    print(f"File: {results['file_path']}")
    print(f"Size: {results['file_size'] / (1024*1024):.2f} MB")
    print(f"Groups: {structure['group_count']}")
    print(f"Datasets: {structure['dataset_count']}")
    print(f"Max depth: {structure['max_depth']}")
    
    # Print group name patterns
    if structure["group_name_patterns"]:
        print("\n=== GROUP NAME PATTERNS ===")
        for prefix, count, min_num, max_num in structure["group_name_patterns"]:
            print(f"Pattern: '{prefix}N' - {count} groups with N ranging from {min_num} to {max_num}")
    
    # Print large datasets
    if structure["large_datasets"]:
        print("\n=== LARGE DATASETS ===")
        for name, shape, dtype, size in sorted(structure["large_datasets"], 
                                              key=lambda x: x[3], reverse=True)[:10]:
            print(f"{name}: shape={shape}, dtype={dtype}, elements={size:,}")
    
    # Print time series candidates
    if structure["time_series_candidates"]:
        print("\n=== POTENTIAL TIME SERIES DATA ===")
        for name, shape, dtype in structure["time_series_candidates"][:10]:
            print(f"{name}: shape={shape}, dtype={dtype}")
    
    # Print metadata group stats
    if structure["metadata_groups"]:
        print(f"\n=== METADATA GROUPS ===")
        print(f"Found {len(structure['metadata_groups'])} timestamp-based metadata groups")
        
        # Group by parent path
        by_parent = defaultdict(int)
        for path in structure["metadata_groups"]:
            parent = '/'.join(path.split('/')[:-1])
            by_parent[parent] += 1
        
        for parent, count in sorted(by_parent.items(), key=lambda x: x[1], reverse=True):
            print(f"{parent}: {count} metadata entries")
    
    # Print examined datasets
    if results["examined_datasets"]:
        print("\n=== SAMPLE DATASET DETAILS ===")
        for dataset_info in results["examined_datasets"]:
            print(f"\n{dataset_info['name']}:")
            stats = dataset_info["stats"]
            print(f"  Shape: {stats['shape']} ({stats['ndim']} dimensions)")
            print(f"  Type: {stats['dtype']}")
            print(f"  Attributes: {dataset_info['attrs']}")
            
            if "min" in stats:
                print(f"  Range: {stats['min']} to {stats['max']}")
                print(f"  Mean: {stats['mean']:.4f}, Std Dev: {stats['std']:.4f}")
                if stats["has_nans"]:
                    print("  Note: Contains NaN values")
                if stats["has_zeros"]:
                    print("  Note: Contains zero values")
    
    # Print examined groups
    if results["examined_groups"]:
        print("\n=== SAMPLE METADATA GROUP DETAILS ===")
        for group_info in results["examined_groups"]:
            print(f"\n{group_info['name']}:")
            print(f"  Attributes: {group_info['attrs']}")
            if group_info['contents']:
                print(f"  Contents: {len(group_info['contents'])} items")
                for name, type_name in group_info['contents'][:3]:
                    print(f"    - {name}: {type_name}")
                if len(group_info['contents']) > 3:
                    print(f"    - ... and {len(group_info['contents']) - 3} more items")

def print_structure(h5file: h5py.File, max_groups: int = 10, max_datasets: int = 5, 
                   max_depth: Optional[int] = None, path: str = "/") -> None:
    """
    Print the structure of an HDF5 file with limits for better readability
    
    Args:
        h5file: The HDF5 file to explore
        max_groups: Maximum number of groups to show at each level
        max_datasets: Maximum number of datasets to show for each group
        max_depth: Maximum depth level to display
        path: Current path in the file
    """
    try:
        obj = h5file[path] if path != "/" else h5file
    except KeyError:
        print(f"Path {path} not found in file")
        return
    
    # Determine current depth
    depth = path.count('/') if path != "/" else 0
    
    # Stop if we've reached max depth
    if max_depth is not None and depth >= max_depth:
        return
    
    # Print current path
    indent = "  " * depth
    path_name = path if path != "/" else "/"
    print(f"{indent}{path_name}")
    
    if isinstance(obj, h5py.Group):
        # Separate datasets and groups
        groups = []
        datasets = []
        
        for name, child in obj.items():
            full_path = f"{path}/{name}" if path != "/" else f"/{name}"
            if isinstance(child, h5py.Group):
                groups.append((name, full_path))
            elif isinstance(child, h5py.Dataset):
                datasets.append((name, child))
        
        # Show number of groups and datasets
        if groups:
            disp_groups = min(len(groups), max_groups)
            print(f"{indent}  [Groups: {len(groups)} total, showing {disp_groups}]")
            
            # Print groups
            for i, (name, full_path) in enumerate(sorted(groups)):
                if i >= max_groups:
                    break
                print_structure(h5file, max_groups, max_datasets, max_depth, full_path)
        
        # Show datasets
        if datasets:
            disp_datasets = min(len(datasets), max_datasets)
            print(f"{indent}  [Datasets: {len(datasets)} total, showing {disp_datasets}]")
            
            # Print datasets
            for i, (name, dataset) in enumerate(sorted(datasets)):
                if i >= max_datasets:
                    break
                
                shape_str = " × ".join(str(dim) for dim in dataset.shape)
                type_str = str(dataset.dtype)
                attr_count = len(dataset.attrs)
                
                print(f"{indent}  {name}: shape=({shape_str}), type={type_str}, {attr_count} attributes")

def main() -> None:
    """Main function for the HDF5 Inspector Abridged"""
    parser = argparse.ArgumentParser(description="Abridged inspector for HDF5 files")
    parser.add_argument("file", help="Path to the HDF5 file to inspect")
    parser.add_argument("--max-groups", type=int, default=10,
                      help="Maximum number of groups to show at each level (default: 10)")
    parser.add_argument("--max-datasets", type=int, default=5,
                      help="Maximum number of datasets to show for each group (default: 5)")
    parser.add_argument("--max-depth", type=int, default=None,
                      help="Maximum depth level to display (default: unlimited)")
    parser.add_argument("--path", default="/",
                      help="Path within the HDF5 file to start inspection (default: /)")
    parser.add_argument("--summary-only", action="store_true",
                      help="Only show summary statistics, not the detailed structure")
    parser.add_argument("--examine-large", action="store_true",
                      help="Examine details of large datasets")
    parser.add_argument("--examine-metadata", action="store_true",
                      help="Examine details of metadata groups")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.file):
        print(f"Error: File {args.file} not found")
        return
    
    analysis_options = {
        "examine_large_datasets": args.examine_large,
        "examine_metadata": args.examine_metadata,
        "max_datasets": args.max_datasets,
        "max_groups": args.max_groups,
    }
    
    # Analyze the file
    results = analyze_file(args.file, analysis_options)
    
    if "error" in results:
        print(f"Error analyzing file: {results['error']}")
        return
    
    # Print summary
    print_file_summary(results)
    
    # Print structure if requested
    if not args.summary_only:
        try:
            with h5py.File(args.file, 'r') as f:
                print("\n=== HDF5 FILE STRUCTURE ===")
                print_structure(f, args.max_groups, args.max_datasets, args.max_depth, args.path)
        except Exception as e:
            print(f"Error displaying file structure: {e}")

if __name__ == "__main__":
    main()