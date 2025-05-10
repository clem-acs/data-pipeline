#!/usr/bin/env python3
"""
Debug script for examining H5 files created by the event transform.

Usage:
    python debug.py path-to-file.h5
"""

import sys
import os
import h5py
import json
import numpy as np
import traceback
from typing import Dict, List, Any, Optional, Tuple


def safe_str(value):
    """Safely convert value to string, handling bytes."""
    if value is None:
        return "None"
    if isinstance(value, bytes):
        try:
            return value.decode('utf-8')
        except:
            return f"[Bytes: {len(value)}]"
    if isinstance(value, np.ndarray):
        return f"[Array: {value.shape}]"
    return str(value)


def safe_print(prefix, value):
    """Print value with prefix, handling exceptions."""
    try:
        print(f"{prefix}: {safe_str(value)}")
    except:
        print(f"{prefix}: [Error printing value]")


def print_section(title):
    """Print a section divider."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def inspect_file_structure(h5_file):
    """Print the overall structure of the H5 file."""
    print_section("FILE STRUCTURE")
    
    def print_group_structure(group, indent=0):
        """Recursively print group structure."""
        for key in group.keys():
            item = group[key]
            indent_str = "  " * indent
            
            if isinstance(item, h5py.Group):
                print(f"{indent_str}Group: {key}/")
                print_group_structure(item, indent + 1)
            elif isinstance(item, h5py.Dataset):
                try:
                    shape = item.shape
                    dtype = item.dtype
                    print(f"{indent_str}Dataset: {key} (Shape: {shape}, Type: {dtype})")
                except:
                    print(f"{indent_str}Dataset: {key} [Error reading metadata]")
    
    try:
        print_group_structure(h5_file)
    except Exception as e:
        print(f"Error inspecting file structure: {e}")
        traceback.print_exc()


def inspect_elements(h5_file):
    """Inspect the elements table, handling potential errors."""
    print_section("ELEMENTS INSPECTION")
    
    if 'elements' not in h5_file:
        print("No elements group found!")
        return
    
    if 'table' not in h5_file['elements']:
        print("No elements table dataset found!")
        return
    
    elements_table = h5_file['elements']['table']
    
    print(f"Elements table shape: {elements_table.shape}")
    print(f"Elements table dtype: {elements_table.dtype}")
    
    # Try multiple approaches to access elements data
    try:
        print("\nApproach 1: Direct read of first element")
        first_element = elements_table[0]
        for field in elements_table.dtype.names:
            safe_print(f"  {field}", first_element[field])
    except Exception as e:
        print(f"Error in Approach 1: {e}")
    
    try:
        print("\nApproach 2: Field-by-field read")
        for field in elements_table.dtype.names:
            try:
                value = elements_table[0][field]
                safe_print(f"  {field}", value)
            except Exception as e:
                print(f"  {field}: [Error: {e}]")
    except Exception as e:
        print(f"Error in Approach 2: {e}")
    
    try:
        print("\nApproach 3: Numpy array conversion")
        # Try to convert the first element to a numpy array
        element_array = np.array(elements_table[0:1])
        print(f"  Converted to array with shape: {element_array.shape}")
    except Exception as e:
        print(f"Error in Approach 3: {e}")
    
    # Try to identify null issues
    print("\nNull check: Scanning for critical fields with None/empty values")
    null_fields = {}
    
    for field in ['element_id', 'task_id', 'task_type']:
        if field not in elements_table.dtype.names:
            print(f"  Field {field} not in dtype!")
            continue
            
        try:
            for i in range(min(10, len(elements_table))):
                try:
                    value = elements_table[i][field]
                    if value is None or (isinstance(value, (str, bytes)) and len(value) == 0):
                        if field not in null_fields:
                            null_fields[field] = []
                        null_fields[field].append(i)
                except:
                    if field not in null_fields:
                        null_fields[field] = []
                    null_fields[field].append(i)
        except Exception as e:
            print(f"  Error checking nulls in {field}: {e}")
    
    if null_fields:
        print("  Found null/empty values in critical fields:")
        for field, indices in null_fields.items():
            print(f"    {field}: {len(indices)} nulls, first few at indices {indices[:5]}")
    else:
        print("  No null/empty values found in sampled elements (first 10)")


def inspect_tasks(h5_file):
    """Inspect the tasks table, handling potential errors."""
    print_section("TASKS INSPECTION")
    
    if 'tasks' not in h5_file:
        print("No tasks group found!")
        return
    
    if 'table' not in h5_file['tasks']:
        print("No tasks table dataset found!")
        return
    
    tasks_table = h5_file['tasks']['table']
    
    print(f"Tasks table shape: {tasks_table.shape}")
    print(f"Tasks table dtype: {tasks_table.dtype}")
    
    # Similar approaches to elements
    try:
        print("\nApproach 1: Direct read of first task")
        first_task = tasks_table[0]
        for field in tasks_table.dtype.names:
            safe_print(f"  {field}", first_task[field])
    except Exception as e:
        print(f"Error in Approach 1: {e}")
    
    try:
        print("\nApproach 2: Field-by-field read")
        for field in tasks_table.dtype.names:
            try:
                value = tasks_table[0][field]
                safe_print(f"  {field}", value)
            except Exception as e:
                print(f"  {field}: [Error: {e}]")
    except Exception as e:
        print(f"Error in Approach 2: {e}")
    
    # Check for null task_ids
    print("\nNull check: Scanning for null/empty task_ids")
    
    null_task_ids = []
    
    try:
        for i in range(len(tasks_table)):
            try:
                task_id = tasks_table[i]['task_id']
                if task_id is None or (isinstance(task_id, (str, bytes)) and len(task_id) == 0):
                    null_task_ids.append(i)
            except:
                null_task_ids.append(i)
    except Exception as e:
        print(f"  Error checking null task_ids: {e}")
    
    if null_task_ids:
        print(f"  Found {len(null_task_ids)} tasks with null/empty task_ids")
    else:
        print("  No null/empty task_ids found")
    
    # Check element_ids array field
    print("\nElement IDs check: Examine element_ids arrays")
    
    try:
        if 'element_ids' in tasks_table.dtype.names:
            empty_element_lists = []
            
            for i in range(len(tasks_table)):
                try:
                    element_ids = tasks_table[i]['element_ids']
                    if element_ids is None or len(element_ids) == 0:
                        empty_element_lists.append(i)
                except:
                    empty_element_lists.append(i)
            
            print(f"  Tasks with empty element_ids: {len(empty_element_lists)} out of {len(tasks_table)}")
        else:
            print("  No 'element_ids' field found in tasks table")
    except Exception as e:
        print(f"  Error checking element_ids: {e}")


def inspect_segments(h5_file):
    """Inspect segments data."""
    print_section("SEGMENTS INSPECTION")
    
    if 'segments' not in h5_file:
        print("No segments group found!")
        return
    
    segment_types = ['recording', 'thinking', 'pause']
    
    for segment_type in segment_types:
        print(f"\nChecking {segment_type} segments:")
        
        if segment_type not in h5_file['segments']:
            print(f"  No {segment_type} dataset found")
            continue
        
        segments = h5_file['segments'][segment_type]
        print(f"  Shape: {segments.shape}")
        print(f"  Dtype: {segments.dtype}")
        
        if len(segments) == 0:
            print("  Empty dataset")
            continue
        
        # Check container references
        try:
            containing_elements = []
            containing_tasks = []
            
            for i in range(len(segments)):
                try:
                    element_id = segments[i]['containing_element_id']
                    if element_id is not None and (isinstance(element_id, (str, bytes)) and len(element_id) > 0):
                        containing_elements.append(i)
                except:
                    pass
                    
                try:
                    task_id = segments[i]['containing_task_id']
                    if task_id is not None and (isinstance(task_id, (str, bytes)) and len(task_id) > 0):
                        containing_tasks.append(i)
                except:
                    pass
            
            print(f"  Segments with element container: {len(containing_elements)} out of {len(segments)}")
            print(f"  Segments with task container: {len(containing_tasks)} out of {len(segments)}")
            print(f"  Segments with no container: {len(segments) - len(containing_elements) - len(containing_tasks)}")
        except Exception as e:
            print(f"  Error checking containers: {e}")


def inspect_indices(h5_file):
    """Inspect index data."""
    print_section("INDEX INSPECTION")
    
    if 'indices' not in h5_file:
        print("No indices group found!")
        return
    
    indices_group = h5_file['indices']
    print("Found indices group with:")
    
    for index_type in indices_group.keys():
        print(f"\nIndex type: {index_type}")
        index_group = indices_group[index_type]
        
        print(f"  Contains {len(index_group.keys())} entries:")
        
        key_sample = list(index_group.keys())[:5]  # Show just a few for brevity
        print(f"  Sample keys: {key_sample}...")
        
        # Pick the first index and show content if it exists
        if len(key_sample) > 0:
            first_key = key_sample[0]
            try:
                index_data = index_group[first_key]
                print(f"  Sample data for '{first_key}':")
                print(f"    Shape: {index_data.shape}")
                print(f"    Type: {index_data.dtype}")
                
                if len(index_data) > 0:
                    try:
                        first_value = index_data[0]
                        print(f"    First value: {safe_str(first_value)}")
                    except Exception as e:
                        print(f"    Error accessing first value: {e}")
            except Exception as e:
                print(f"  Error inspecting index '{first_key}': {e}")


def verify_relationships(h5_file):
    """Verify relationships between entities."""
    print_section("RELATIONSHIP VERIFICATION")
    
    # Check if we have all required components
    if not all(key in h5_file for key in ['elements', 'tasks', 'segments', 'indices']):
        missing = [key for key in ['elements', 'tasks', 'segments', 'indices'] if key not in h5_file]
        print(f"Missing key groups: {missing}")
        return
    
    # Verify task-element links
    print("\nVerifying task-element links:")
    try:
        tasks_table = h5_file['tasks']['table']
        elements_table = h5_file['elements']['table']
        
        task_count = len(tasks_table)
        element_count = len(elements_table)
        
        print(f"  Found {task_count} tasks and {element_count} elements")
        
        # Check if element_by_task index matches actual task data
        if 'element_by_task' in h5_file['indices']:
            element_by_task = h5_file['indices']['element_by_task']
            indexed_tasks = list(element_by_task.keys())
            
            print(f"  Tasks with element indexes: {len(indexed_tasks)}")
            
            # Compare with tasks having element_ids
            tasks_with_elements = 0
            for i in range(task_count):
                try:
                    element_ids = tasks_table[i]['element_ids']
                    if element_ids is not None and len(element_ids) > 0:
                        tasks_with_elements += 1
                except:
                    pass
            
            print(f"  Tasks with element_ids array: {tasks_with_elements}")
            
            # Check consistency
            print(f"  Consistency check: {tasks_with_elements} tasks with elements vs {len(indexed_tasks)} indexed")
        else:
            print("  No element_by_task index found")
            
    except Exception as e:
        print(f"  Error verifying task-element links: {e}")
        traceback.print_exc()
    
    # Verify segment-container links
    print("\nVerifying segment-container links:")
    try:
        # Check if segments_by_element index matches
        if 'segments_by_element' in h5_file['indices']:
            segments_by_element = h5_file['indices']['segments_by_element']
            elements_with_segments = len(segments_by_element.keys())
            
            print(f"  Elements with segment indexes: {elements_with_segments}")
        else:
            print("  No segments_by_element index found")
            
        # Check if segments_by_task index matches
        if 'segments_by_task' in h5_file['indices']:
            segments_by_task = h5_file['indices']['segments_by_task']
            tasks_with_segments = len(segments_by_task.keys())
            
            print(f"  Tasks with segment indexes: {tasks_with_segments}")
        else:
            print("  No segments_by_task index found")
    except Exception as e:
        print(f"  Error verifying segment links: {e}")


def inspect_null_pointers(h5_file):
    """Advanced null pointer inspection."""
    print_section("NULL POINTER ANALYSIS")
    
    # Check for VLEN fields that might cause null pointer issues
    print("\nVLEN field inspection:")
    
    def check_vlen_fields(dataset, name):
        """Check a dataset for VLEN fields that might have nulls."""
        try:
            dtype = dataset.dtype
            vlen_fields = []
            
            # Find VLEN fields
            for field in dtype.names:
                if h5py.check_dtype(vlen=dtype[field]) is not None:
                    vlen_fields.append(field)
            
            if not vlen_fields:
                print(f"  {name}: No VLEN fields found")
                return
                
            print(f"  {name}: Found VLEN fields: {vlen_fields}")
            
            # Try to read each VLEN field directly
            for field in vlen_fields:
                print(f"    Testing direct field access for '{field}':")
                try:
                    # Try different access patterns
                    try:
                        # First approach: read entire field
                        field_data = dataset[field]
                        print(f"      Read all data: {len(field_data)} items")
                    except Exception as e1:
                        print(f"      Failed to read all data: {e1}")
                        
                        # Second approach: read item by item
                        success = 0
                        errors = 0
                        null_values = 0
                        
                        for i in range(min(len(dataset), 20)):  # Sample first 20
                            try:
                                value = dataset[i][field]
                                success += 1
                                if value is None:
                                    null_values += 1
                            except:
                                errors += 1
                        
                        print(f"      Item-by-item: {success} successes, {errors} errors, {null_values} nulls")
                except Exception as e:
                    print(f"      Error in field checking: {e}")
        except Exception as e:
            print(f"  Error checking {name}: {e}")
    
    # Check main data tables
    if 'elements' in h5_file and 'table' in h5_file['elements']:
        check_vlen_fields(h5_file['elements']['table'], "Elements table")
    
    if 'tasks' in h5_file and 'table' in h5_file['tasks']:
        check_vlen_fields(h5_file['tasks']['table'], "Tasks table")
    
    for segment_type in ['recording', 'thinking', 'pause']:
        if 'segments' in h5_file and segment_type in h5_file['segments']:
            check_vlen_fields(h5_file['segments'][segment_type], f"{segment_type} segments")


def try_read_data_directly(h5_file, path, max_items=5):
    """Try to read data directly using various methods."""
    print(f"\nAttempting to read: {path}")
    
    if path not in h5_file:
        print("  Path not found!")
        return
    
    dataset = h5_file[path]
    print(f"  Dataset shape: {dataset.shape}")
    print(f"  Dataset type: {dataset.dtype}")
    
    # Method 1: Direct read
    try:
        print("  Method 1: Direct read with [:]")
        data = dataset[:]
        print(f"    Success, got array of shape {data.shape}")
    except Exception as e:
        print(f"    Failed: {e}")
    
    # Method 2: Field-by-field read for compound datatypes
    if hasattr(dataset.dtype, 'names') and dataset.dtype.names:
        print("  Method 2: Field-by-field read")
        for field in dataset.dtype.names:
            try:
                if len(dataset) > 0:
                    value = dataset[0][field]
                    print(f"    Field '{field}' first value: {safe_str(value)}")
            except Exception as e:
                print(f"    Failed for field '{field}': {e}")
    
    # Method 3: Item-by-item read
    try:
        print("  Method 3: Item-by-item read")
        for i in range(min(len(dataset), max_items)):
            try:
                item = dataset[i]
                print(f"    Item {i}: {type(item)}")
            except Exception as e:
                print(f"    Failed for item {i}: {e}")
    except Exception as e:
        print(f"    Failed to iterate: {e}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python debug.py path-to-file.h5")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        sys.exit(1)
    
    print(f"Analyzing file: {file_path}")
    
    try:
        with h5py.File(file_path, 'r') as h5_file:
            # File info
            print_section("FILE INFO")
            print(f"File path: {file_path}")
            print(f"File size: {os.path.getsize(file_path) / (1024*1024):.2f} MB")
            
            # Run inspections
            inspect_file_structure(h5_file)
            inspect_tasks(h5_file)
            inspect_elements(h5_file)
            inspect_segments(h5_file)
            inspect_indices(h5_file)
            verify_relationships(h5_file)
            inspect_null_pointers(h5_file)
            
            # Try direct reads of problematic datasets
            print_section("DIRECT READ ATTEMPTS")
            problematic_paths = [
                'elements/table',
                'tasks/table',
                'segments/recording',
                'segments/thinking'
            ]
            
            for path in problematic_paths:
                try_read_data_directly(h5_file, path)
    except Exception as e:
        print(f"Error opening H5 file: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()