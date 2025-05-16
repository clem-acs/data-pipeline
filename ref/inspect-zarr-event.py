#!/usr/bin/env python3
"""
Zarr Event File Inspector

This script inspects a Zarr event file (created by t2C_event_v0 transform) and displays detailed information
about its structure, contents, and relationships.

Usage:
    python inspect-zarr-event.py <path-to-zarr.zarr>
"""

import sys
import os
import json
import argparse
import numpy as np
import pandas as pd
import xarray as xr
import zarr
from datetime import datetime
import matplotlib.pyplot as plt
from tabulate import tabulate
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)

def print_section_header(title):
    """Print a formatted section header"""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80 + "\n")

def print_subsection_header(title):
    """Print a formatted subsection header"""
    print("\n" + "-" * 80)
    print(f" {title} ".center(80, "-"))
    print("-" * 80 + "\n")

def get_zarr_store_info(zarr_path):
    """Get low-level information about the Zarr store"""
    print_subsection_header("Zarr Store Structure (Low Level)")
    
    try:
        # Open the zarr store
        store = zarr.open(zarr_path, mode='r')
        
        # Print zarr version
        print(f"Zarr Version: {zarr.__version__}")
        
        # Determine Zarr format (v2 or v3)
        zarr_version = 3
        try:
            # Try to access a v2-specific attribute
            next(iter(store.values())).compressor
            zarr_version = 2
        except (StopIteration, AttributeError, TypeError):
            zarr_version = 3
        
        print(f"Zarr Format Version: {zarr_version}")
        
        # Print store attributes
        print(f"Store Attributes: {dict(store.attrs)}")
        
        # List all arrays in the store
        print("\nZarr Arrays:")
        for name in sorted(store):
            if name != '.zattrs' and name != '.zgroup':
                try:
                    array = store[name]
                    print(f"  - {name}:")
                    print(f"      Shape: {array.shape}")
                    print(f"      Chunks: {array.chunks}")
                    print(f"      Dtype: {array.dtype}")
                    
                    # Only show these attributes for Zarr v2
                    if zarr_version == 2:
                        try:
                            print(f"      Compressor: {array.compressor}")
                            print(f"      Filters: {array.filters}")
                            print(f"      Order: {array.order}")
                        except (AttributeError, TypeError):
                            pass
                    
                    print(f"      Fill Value: {array.fill_value}")
                    print(f"      Size: {array.nbytes / (1024 * 1024):.2f} MB")
                    
                    # Print sample values if appropriate
                    if array.size > 0 and array.shape[0] > 0:
                        sample_count = min(5, array.shape[0])
                        try:
                            print(f"      Sample Values (first {sample_count}):")
                            sample = array[:sample_count]
                            for i, value in enumerate(sample):
                                print(f"        [{i}]: {value}")
                        except Exception as e:
                            print(f"      Error getting sample values: {e}")
                except Exception as e:
                    print(f"  - {name}: Error accessing array: {e}")
        
        # Print all metadata files
        print("\nZarr Metadata Files:")
        for name in ['.zattrs', '.zgroup']:
            if name in store:
                try:
                    meta = store[name]
                    print(f"  - {name}: {meta}")
                except Exception as e:
                    print(f"  - {name}: Error: {e}")
    
    except Exception as e:
        print(f"Error accessing Zarr store: {e}")

def analyze_dataset(ds):
    """Analyze and display detailed information about the xarray Dataset"""
    print_subsection_header("Dataset Overview")
    
    # Basic dataset info
    print(f"Session ID: {ds.attrs.get('session_id', 'Unknown')}")
    print(f"Transform: {ds.attrs.get('transform', 'Unknown')}")
    print(f"Version: {ds.attrs.get('version', 'Unknown')}")
    print(f"Created At: {ds.attrs.get('created_at', 'Unknown')}")
    print(f"Is Empty: {ds.attrs.get('empty', False)}")
    
    # Print dimensions
    print("\nDimensions:")
    for dim_name, dim_size in ds.dims.items():
        print(f"  - {dim_name}: {dim_size}")
    
    # Print coordinates
    print("\nCoordinates:")
    for coord_name, coord_var in ds.coords.items():
        print(f"  - {coord_name}:")
        print(f"      Shape: {coord_var.shape}")
        print(f"      Dtype: {coord_var.dtype}")
        print(f"      Chunks: {coord_var.chunks}")
        print(f"      Encoding: {coord_var.encoding}")
        
        # Print sample values
        sample_count = min(5, coord_var.size)
        if sample_count > 0:
            sample_values = coord_var.values[:sample_count]
            print(f"      Sample Values (first {sample_count}): {sample_values}")
    
    # Print data variables summary
    print("\nData Variables:")
    for var_name, var in ds.data_vars.items():
        print(f"  - {var_name}:")
        print(f"      Dimensions: {var.dims}")
        print(f"      Shape: {var.shape}")
        print(f"      Dtype: {var.dtype}")
        print(f"      Chunks: {var.chunks}")
        
        # Determine variable category based on prefix
        if var_name.startswith('task_'):
            category = 'task'
        elif var_name.startswith('element_'):
            category = 'element'
        elif var_name.startswith('segment_'):
            category = 'segment'
        else:
            category = 'other'
            
        print(f"      Category: {category}")
        
        # Print sample values for small arrays
        if var.size < 100:  # Only show for manageable sizes
            try:
                sample_count = min(5, var.shape[0])
                if sample_count > 0:
                    sample_values = var.values[:sample_count]
                    print(f"      Sample Values (first {sample_count}): {sample_values}")
            except Exception as e:
                print(f"      Error getting sample values: {e}")

def analyze_tasks(ds):
    """Analyze and display information about tasks"""
    print_subsection_header("Tasks Analysis")
    
    # Check if we have task data
    if 'task_id' not in ds.coords:
        print("No task data found in the dataset.")
        return
    
    task_ids = ds.coords['task_id'].values
    task_count = len(task_ids)
    
    print(f"Total Tasks: {task_count}")
    
    # Get task fields
    task_fields = [v for v in ds.data_vars if v.startswith('task_')]
    
    if task_count > 0 and len(task_fields) > 0:
        # Create a DataFrame with task data
        task_data = {}
        for field in task_fields:
            # Drop the 'task_' prefix for cleaner display
            field_name = field[5:]
            task_data[field_name] = ds[field].values
            
        df = pd.DataFrame(task_data, index=task_ids)
        df.index.name = 'task_id'
        
        print("\nTask Summary (first 10 tasks):")
        pd.set_option('display.max_columns', None)
        pd.set_option('display.expand_frame_repr', False)
        print(df.head(10))
        
        # Check for completed vs skipped tasks
        if 'task_completion_status' in task_fields:
            status_counts = pd.Series(ds['task_completion_status'].values).value_counts()
            print("\nTask Completion Status Counts:")
            print(status_counts)
        
        # Count by task type
        if 'task_task_type' in task_fields:
            type_counts = pd.Series(ds['task_task_type'].values).value_counts()
            print("\nTask Type Counts:")
            print(type_counts)
            
            # Plot task type distribution if there are multiple types
            if len(type_counts) > 1:
                plt.figure(figsize=(10, 6))
                type_counts.plot(kind='bar')
                plt.title('Task Type Distribution')
                plt.ylabel('Count')
                plt.xlabel('Task Type')
                plt.tight_layout()
                plt.savefig('task_type_distribution.png')
                print("\nTask type distribution plot saved to 'task_type_distribution.png'")
        
        # Task duration statistics
        if 'task_duration' in task_fields:
            durations = ds['task_duration'].values
            print("\nTask Duration Statistics (seconds):")
            print(f"  Min: {np.min(durations):.2f}")
            print(f"  Max: {np.max(durations):.2f}")
            print(f"  Mean: {np.mean(durations):.2f}")
            print(f"  Median: {np.median(durations):.2f}")
            print(f"  Std Dev: {np.std(durations):.2f}")
            print(f"  Total: {np.sum(durations):.2f}")

def analyze_elements(ds):
    """Analyze and display information about elements"""
    print_subsection_header("Elements Analysis")
    
    # Check if we have element data
    if 'element_id' not in ds.coords:
        print("No element data found in the dataset.")
        return
    
    element_ids = ds.coords['element_id'].values
    element_count = len(element_ids)
    
    print(f"Total Elements: {element_count}")
    
    # Get element fields
    element_fields = [v for v in ds.data_vars if v.startswith('element_')]
    
    if element_count > 0 and len(element_fields) > 0:
        # Create a DataFrame with sample element data (first 10)
        sample_size = min(10, element_count)
        sample_data = {}
        for field in element_fields:
            # Drop the 'element_' prefix for cleaner display
            field_name = field[8:]
            sample_data[field_name] = ds[field].values[:sample_size]
            
        df = pd.DataFrame(sample_data, index=element_ids[:sample_size])
        df.index.name = 'element_id'
        
        print("\nElement Sample (first 10 elements):")
        pd.set_option('display.max_columns', None)
        pd.set_option('display.expand_frame_repr', False)
        print(df)
        
        # Element type distribution
        if 'element_element_type' in element_fields:
            type_counts = pd.Series(ds['element_element_type'].values).value_counts()
            print("\nElement Type Counts:")
            print(type_counts)
        
        # Elements per task
        if 'element_task_id' in element_fields:
            task_element_counts = pd.Series(ds['element_task_id'].values).value_counts()
            print("\nElements per Task (Top 10):")
            print(task_element_counts.head(10))
        
        # Element duration statistics
        if 'element_duration' in element_fields:
            durations = ds['element_duration'].values
            print("\nElement Duration Statistics (seconds):")
            print(f"  Min: {np.min(durations):.2f}")
            print(f"  Max: {np.max(durations):.2f}")
            print(f"  Mean: {np.mean(durations):.2f}")
            print(f"  Median: {np.median(durations):.2f}")
            print(f"  Std Dev: {np.std(durations):.2f}")
            
        # Check for instruction elements
        if 'element_is_instruction' in element_fields:
            instruction_count = np.sum(ds['element_is_instruction'].values)
            print(f"\nInstruction Elements: {instruction_count} ({instruction_count/element_count*100:.2f}%)")
            
        # Audio mode distribution
        if 'element_audio_mode' in element_fields:
            audio_mode_counts = pd.Series(ds['element_audio_mode'].values).value_counts()
            print("\nAudio Mode Distribution:")
            print(audio_mode_counts)

def analyze_segments(ds):
    """Analyze and display information about segments"""
    print_subsection_header("Segments Analysis")
    
    # Check if we have segment data
    if 'segment_id' not in ds.coords:
        print("No segment data found in the dataset.")
        return
    
    segment_ids = ds.coords['segment_id'].values
    segment_count = len(segment_ids)
    
    print(f"Total Segments: {segment_count}")
    
    # Get segment fields
    segment_fields = [v for v in ds.data_vars if v.startswith('segment_')]
    
    if segment_count > 0 and len(segment_fields) > 0:
        # Create a DataFrame with sample segment data (first 10)
        sample_size = min(10, segment_count)
        sample_data = {}
        for field in segment_fields:
            # Drop the 'segment_' prefix for cleaner display
            field_name = field[8:]
            sample_data[field_name] = ds[field].values[:sample_size]
            
        df = pd.DataFrame(sample_data, index=segment_ids[:sample_size])
        df.index.name = 'segment_id'
        
        print("\nSegment Sample (first 10 segments):")
        pd.set_option('display.max_columns', None)
        pd.set_option('display.expand_frame_repr', False)
        print(df)
        
        # Segment type distribution
        if 'segment_segment_type' in segment_fields:
            type_counts = pd.Series(ds['segment_segment_type'].values).value_counts()
            print("\nSegment Type Counts:")
            print(type_counts)
            
            # Plot segment type distribution if there are multiple types
            if len(type_counts) > 1:
                plt.figure(figsize=(10, 6))
                type_counts.plot(kind='bar')
                plt.title('Segment Type Distribution')
                plt.ylabel('Count')
                plt.xlabel('Segment Type')
                plt.tight_layout()
                plt.savefig('segment_type_distribution.png')
                print("\nSegment type distribution plot saved to 'segment_type_distribution.png'")
        
        # Segments per element
        if 'segment_containing_element_id' in segment_fields:
            # Filter out empty element IDs
            containing_elements = [e for e in ds['segment_containing_element_id'].values if e]
            if containing_elements:
                element_segment_counts = pd.Series(containing_elements).value_counts()
                print("\nSegments per Element (Top 10):")
                print(element_segment_counts.head(10))
            else:
                print("\nNo segments are associated with elements.")
        
        # Segment duration statistics
        if 'segment_duration' in segment_fields:
            durations = ds['segment_duration'].values
            print("\nSegment Duration Statistics (seconds):")
            print(f"  Min: {np.min(durations):.2f}")
            print(f"  Max: {np.max(durations):.2f}")
            print(f"  Mean: {np.mean(durations):.2f}")
            print(f"  Median: {np.median(durations):.2f}")
            print(f"  Std Dev: {np.std(durations):.2f}")

def analyze_relationships(ds):
    """Analyze and display relationship information between entities"""
    print_subsection_header("Entity Relationships")
    
    # Check for task-element relationships
    task_element_mapping = {}
    
    if 'element_task_id' in ds.data_vars and 'element_id' in ds.coords:
        element_ids = ds.coords['element_id'].values
        task_ids = ds['element_task_id'].values
        
        # Build mapping from task_id to element_ids
        for i, task_id in enumerate(task_ids):
            if not task_id:  # Skip empty task IDs
                continue
                
            if task_id not in task_element_mapping:
                task_element_mapping[task_id] = []
                
            task_element_mapping[task_id].append(element_ids[i])
        
        print("Task-Element Relationships:")
        print(f"Tasks with elements: {len(task_element_mapping)}")
        
        # Print sample of task-element mapping
        sample_tasks = list(task_element_mapping.keys())[:5]  # Take first 5 tasks
        print("\nSample task-element mappings:")
        for task_id in sample_tasks:
            element_count = len(task_element_mapping[task_id])
            print(f"  Task '{task_id}' has {element_count} elements")
            if element_count <= 10:
                print(f"    Elements: {task_element_mapping[task_id]}")
            else:
                print(f"    First 5 elements: {task_element_mapping[task_id][:5]}")
    
    # Check for element-segment relationships
    element_segment_mapping = {}
    
    if 'segment_containing_element_id' in ds.data_vars and 'segment_id' in ds.coords:
        segment_ids = ds.coords['segment_id'].values
        element_ids = ds['segment_containing_element_id'].values
        
        # Build mapping from element_id to segment_ids
        for i, element_id in enumerate(element_ids):
            if not element_id:  # Skip empty element IDs
                continue
                
            if element_id not in element_segment_mapping:
                element_segment_mapping[element_id] = []
                
            element_segment_mapping[element_id].append(segment_ids[i])
        
        print("\nElement-Segment Relationships:")
        print(f"Elements with segments: {len(element_segment_mapping)}")
        
        # Print sample of element-segment mapping
        sample_elements = list(element_segment_mapping.keys())[:5]  # Take first 5 elements
        print("\nSample element-segment mappings:")
        for element_id in sample_elements:
            segment_count = len(element_segment_mapping[element_id])
            print(f"  Element '{element_id}' has {segment_count} segments")
            if segment_count <= 10:
                print(f"    Segments: {element_segment_mapping[element_id]}")
            else:
                print(f"    First 5 segments: {element_segment_mapping[element_id][:5]}")
    
    # Check for direct attributes in the dataset that map relationships
    element_segment_attrs = [attr for attr in ds.attrs.keys() if attr.startswith('element_') and attr.endswith('_segments')]
    if element_segment_attrs:
        print("\nExplicit Element-Segment Mappings from attributes:")
        print(f"Mappings defined in attributes: {len(element_segment_attrs)}")
        
        # Print sample of attribute-defined mappings
        sample_attrs = element_segment_attrs[:5]  # Take first 5 attributes
        for attr in sample_attrs:
            element_id = attr[8:-9]  # Remove 'element_' prefix and '_segments' suffix
            segments = ds.attrs[attr]
            print(f"  Element '{element_id}' maps to segments: {segments[:5]} ...")

def analyze_data_types(ds):
    """Analyze and display information about data types in the dataset"""
    print_subsection_header("Data Type Analysis")
    
    # Collect all data types
    data_types = {}
    
    # Check coordinates
    for coord_name, coord_var in ds.coords.items():
        dtype = str(coord_var.dtype)
        if dtype not in data_types:
            data_types[dtype] = []
        data_types[dtype].append(f"coord:{coord_name}")
    
    # Check data variables
    for var_name, var in ds.data_vars.items():
        dtype = str(var.dtype)
        if dtype not in data_types:
            data_types[dtype] = []
        data_types[dtype].append(f"var:{var_name}")
    
    # Print data type summary
    print("Data Types Used in Dataset:")
    for dtype, variables in sorted(data_types.items()):
        var_count = len(variables)
        print(f"  - {dtype}: {var_count} variables")
        
        # Print sample variables of this type
        sample_count = min(5, var_count)
        if sample_count > 0:
            print(f"    Sample variables: {', '.join(variables[:sample_count])}")
            
            # For string types, show additional information
            if dtype.startswith('<U') or dtype.startswith('|S') or dtype == 'object':
                # Find a variable of this type
                sample_var = variables[0].split(':')[1]
                prefix = variables[0].split(':')[0]
                
                if prefix == 'var':
                    values = ds[sample_var].values
                else:  # coord
                    values = ds.coords[sample_var].values
                
                # Check string lengths
                if len(values) > 0:
                    try:
                        lengths = [len(str(v)) for v in values[:100]]  # Sample first 100
                        print(f"    String length stats (from {sample_var}):")
                        print(f"      Min: {min(lengths)}")
                        print(f"      Max: {max(lengths)}")
                        print(f"      Mean: {sum(lengths)/len(lengths):.2f}")
                    except:
                        print("    Could not analyze string lengths")

def main():
    """Main function to parse arguments and analyze the Zarr file"""
    parser = argparse.ArgumentParser(description='Inspect a Zarr event file')
    parser.add_argument('zarr_path', help='Path to the Zarr event file')
    parser.add_argument('--skip-plots', action='store_true', help='Skip creating plots')
    parser.add_argument('--zarr-only', action='store_true', help='Use only zarr library, not xarray')
    args = parser.parse_args()
    
    zarr_path = args.zarr_path
    
    if not os.path.exists(zarr_path) and not zarr_path.startswith('s3://'):
        print(f"Error: Zarr path '{zarr_path}' does not exist.")
        sys.exit(1)
    
    print_section_header(f"Zarr Event File Analysis: {zarr_path}")
    
    try:
        # First, always get low-level zarr store info which doesn't depend on xarray
        get_zarr_store_info(zarr_path)
        
        if args.zarr_only:
            print("Skipping xarray-based analysis as requested with --zarr-only")
            print_section_header("Analysis Complete")
            return
            
        # Try to open with xarray, handling Zarr v3 format
        print(f"Opening Zarr dataset with xarray from {zarr_path}...")
        try:
            # Some zarr settings to help with compatibility
            import s3fs
            import zarr.storage
            
            # Try to open with xarray
            ds = xr.open_zarr(zarr_path, consolidated=True)
            
            # Get basic dataset info
            analyze_dataset(ds)
            
            # Analyze tasks
            analyze_tasks(ds)
            
            # Analyze elements
            analyze_elements(ds)
            
            # Analyze segments
            analyze_segments(ds)
            
            # Analyze relationships
            analyze_relationships(ds)
            
            # Analyze data types
            analyze_data_types(ds)
            
        except TypeError as e:
            if "compressor" in str(e) and "not available for Zarr format 3" in str(e):
                print("\n" + "!" * 80)
                print("! ZARR V3 FORMAT DETECTED - LIMITED ANALYSIS AVAILABLE                        !")
                print("!" * 80)
                print("\nDetected Zarr format v3, which has limited xarray compatibility.")
                print("The script cannot perform full xarray-based analysis on this format.")
                print("\nTry using a newer version of xarray and zarr, or use --zarr-only flag")
                print("to only use the zarr library for basic inspection.\n")
                
                # Try to extract basic info directly from zarr
                try:
                    # This is simplified info since we can't use xarray
                    z = zarr.open(zarr_path, mode='r')
                    print("\nBasic Zarr v3 Structure:")
                    print(f"Top-level groups/arrays: {list(z.keys())}")
                    print(f"Attributes: {dict(z.attrs)}")
                    
                    # Try to print some array shapes if possible
                    print("\nArray Shapes:")
                    for k in z.keys():
                        if k not in ['.zattrs', '.zgroup']:
                            try:
                                print(f"  {k}: {z[k].shape}")
                            except:
                                pass
                except Exception as inner_e:
                    print(f"Error accessing zarr data directly: {inner_e}")
            else:
                # Re-raise if not the specific v3 format error
                raise
            
    except Exception as e:
        print(f"Error analyzing Zarr dataset: {e}")
        import traceback
        traceback.print_exc()
    
    print_section_header("Analysis Complete")

if __name__ == "__main__":
    main()