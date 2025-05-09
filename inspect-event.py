#!/usr/bin/env python
"""
Inspect event transform H5 output files to verify structure and relationships.

Usage:
    python inspect-event.py path/to/file.h5
"""

import sys
import h5py
import numpy as np
from collections import defaultdict

def decode_if_bytes(value):
    """Decode value if it's bytes, otherwise return as is."""
    if isinstance(value, bytes):
        return value.decode('utf-8')
    elif value is None:
        return ""
    return value

def print_header(title):
    """Print a section header."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

def print_file_structure(h5obj, prefix=""):
    """Recursively print the HDF5 file structure."""
    if isinstance(h5obj, h5py.Group):
        for key in h5obj.keys():
            item = h5obj[key]
            print(f"{prefix}/{key} : {type(item).__name__}")
            if isinstance(item, h5py.Group):
                print_file_structure(item, prefix=f"{prefix}/{key}")
    
    # Print attributes
    if hasattr(h5obj, 'attrs') and len(h5obj.attrs) > 0:
        print(f"{prefix} Attributes:")
        for attr_name, attr_value in h5obj.attrs.items():
            print(f"{prefix}  {attr_name}: {attr_value}")

def safely_get_dataset(h5f, path):
    """Safely get a dataset with error handling."""
    try:
        if path not in h5f:
            return None
            
        dataset = h5f[path]
        
        # Check if it's a dataset and not a group
        if not isinstance(dataset, h5py.Dataset):
            return None
            
        # Try to read the dataset
        try:
            return dataset[:]
        except ValueError as e:
            if "null pointer" in str(e):
                print(f"  WARNING: Dataset {path} has null data. Skipping.")
                return None
            else:
                raise
    except Exception as e:
        print(f"  ERROR accessing {path}: {str(e)}")
        return None

def inspect_metadata(h5f):
    """Inspect and print metadata."""
    print_header("METADATA")
    
    if 'metadata' in h5f:
        metadata = h5f['metadata']
        
        # Print metadata attributes
        if hasattr(metadata, 'attrs'):
            for key in metadata.attrs:
                try:
                    print(f"  {key}: {metadata.attrs[key]}")
                except Exception as e:
                    print(f"  {key}: <Error reading attribute: {str(e)}>")
        
        # Print statistics if available
        if 'stats' in metadata:
            stats = metadata['stats']
            print("\nStatistics:")
            if hasattr(stats, 'attrs'):
                for key in stats.attrs:
                    try:
                        print(f"  {key}: {stats.attrs[key]}")
                    except Exception as e:
                        print(f"  {key}: <Error reading attribute: {str(e)}>")

def inspect_tasks(h5f):
    """Inspect and print task information."""
    print_header("TASKS")
    
    if 'tasks' not in h5f:
        print("No tasks group found in file")
        return
        
    if 'table' not in h5f['tasks']:
        print("No tasks table found in file")
        return
    
    # Try to safely get the dataset
    tasks = safely_get_dataset(h5f, 'tasks/table')
    if tasks is None or len(tasks) == 0:
        print("Tasks table is empty or could not be read")
        return
    
    print(f"Found {len(tasks)} tasks")
    
    # Print task dtype information
    print("\nTask structure:")
    for field in tasks.dtype.names:
        print(f"  Field: {field} (Type: {tasks.dtype[field]})")
    
    # Print summary of task types
    task_types = defaultdict(int)
    try:
        for task in tasks:
            try:
                task_type = decode_if_bytes(task['task_type'])
                task_types[task_type] += 1
            except Exception as e:
                print(f"  Error getting task type: {str(e)}")
    except Exception as e:
        print(f"Error processing tasks: {str(e)}")
    
    print("\nTask types:")
    for task_type, count in task_types.items():
        print(f"  {task_type}: {count}")
    
    # Print details for each task
    print("\nTask details:")
    try:
        for i, task in enumerate(tasks[:5]):  # Limit to first 5 tasks
            try:
                task_id = decode_if_bytes(task['task_id'])
                task_type = decode_if_bytes(task['task_type'])
                
                print(f"\n  Task {i+1}: {task_id} (Type: {task_type})")
                
                # Print all task fields
                for field in task.dtype.names:
                    try:
                        if field in ['task_id', 'task_type']:
                            continue  # Already printed
                            
                        value = task[field]
                        
                        # Handle special fields
                        if 'segments' in field or field == 'element_ids':
                            if hasattr(value, 'size') and value.size > 0:
                                print(f"    {field}: {value.size} items")
                            else:
                                print(f"    {field}: empty")
                        elif isinstance(value, (bytes, np.bytes_)):
                            print(f"    {field}: {decode_if_bytes(value)}")
                        elif isinstance(value, (float, np.float32, np.float64)):
                            print(f"    {field}: {value:.2f}")
                        else:
                            print(f"    {field}: {value}")
                    except Exception as e:
                        print(f"    {field}: <Error: {str(e)}>")
            except Exception as e:
                print(f"  Error processing task {i+1}: {str(e)}")
        
        if len(tasks) > 5:
            print(f"\n  ... and {len(tasks) - 5} more tasks")
    except Exception as e:
        print(f"Error printing task details: {str(e)}")

def inspect_elements(h5f):
    """Inspect and print element information."""
    print_header("ELEMENTS")
    
    if 'elements' not in h5f:
        print("No elements group found in file")
        return
        
    if 'table' not in h5f['elements']:
        print("No elements table found in file")
        return
    
    # Try to safely get the dataset
    elements = safely_get_dataset(h5f, 'elements/table')
    if elements is None or len(elements) == 0:
        print("Elements table is empty or could not be read")
        return
    
    print(f"Found {len(elements)} elements")
    
    # Print element dtype information
    print("\nElement structure:")
    for field in elements.dtype.names:
        print(f"  Field: {field} (Type: {elements.dtype[field]})")
    
    # Group elements by task and type
    elements_by_task = defaultdict(int)
    element_types = defaultdict(int)
    
    try:
        for element in elements:
            try:
                task_id = decode_if_bytes(element['task_id'])
                elements_by_task[task_id] += 1
                
                element_type = decode_if_bytes(element['element_type'])
                element_types[element_type] += 1
            except Exception as e:
                print(f"  Error processing element: {str(e)}")
    except Exception as e:
        print(f"Error summarizing elements: {str(e)}")
    
    # Print summary of elements by task
    print("\nElements by task:")
    for task_id, count in sorted(elements_by_task.items()):
        print(f"  Task '{task_id}': {count} elements")
    
    print("\nElement types:")
    for element_type, count in element_types.items():
        print(f"  {element_type}: {count}")
    
    # Print sample elements
    print("\nSample elements:")
    try:
        for i in range(min(3, len(elements))):
            try:
                element = elements[i]
                element_id = decode_if_bytes(element['element_id'])
                task_id = decode_if_bytes(element['task_id'])
                element_type = decode_if_bytes(element['element_type'])
                
                print(f"\n  Element {i+1}: {element_id}")
                print(f"    Type: {element_type}")
                print(f"    Task: {task_id}")
                
                # Print all element fields
                for field in element.dtype.names:
                    try:
                        if field in ['element_id', 'task_id', 'element_type']:
                            continue  # Already printed
                            
                        value = element[field]
                        
                        # Handle special fields
                        if 'segments' in field:
                            if hasattr(value, 'size') and value.size > 0:
                                print(f"    {field}: {value.size} items")
                            else:
                                print(f"    {field}: empty")
                        elif isinstance(value, (bytes, np.bytes_)):
                            print(f"    {field}: {decode_if_bytes(value)}")
                        elif isinstance(value, (float, np.float32, np.float64)):
                            print(f"    {field}: {value:.2f}")
                        else:
                            print(f"    {field}: {value}")
                    except Exception as e:
                        print(f"    {field}: <Error: {str(e)}>")
            except Exception as e:
                print(f"  Error processing element {i+1}: {str(e)}")
    except Exception as e:
        print(f"Error printing element details: {str(e)}")

def inspect_segments(h5f):
    """Inspect and print segment information."""
    print_header("SEGMENTS")
    
    if 'segments' not in h5f:
        print("No segments found in file")
        return
    
    try:
        segment_types = list(h5f['segments'].keys())
        print(f"Found {len(segment_types)} segment types: {', '.join(segment_types)}")
        
        # Print summary for each segment type
        for segment_type in segment_types:
            try:
                # Safely get the dataset
                segments = safely_get_dataset(h5f, f'segments/{segment_type}')
                if segments is None:
                    print(f"\n{segment_type.capitalize()} segments: <Error reading dataset>")
                    continue
                
                print(f"\n{segment_type.capitalize()} segments: {len(segments)}")
                
                # Print segment dtype information
                print(f"  Segment structure:")
                for field in segments.dtype.names:
                    print(f"    Field: {field} (Type: {segments.dtype[field]})")
                
                # Count containment
                element_contained = 0
                task_contained = 0
                uncontained = 0
                
                try:
                    for segment in segments:
                        try:
                            element_id = decode_if_bytes(segment['containing_element_id'])
                            task_id = decode_if_bytes(segment['containing_task_id'])
                            
                            if element_id:
                                element_contained += 1
                            elif task_id:
                                task_contained += 1
                            else:
                                uncontained += 1
                        except Exception as e:
                            print(f"    Error processing segment: {str(e)}")
                            continue
                except Exception as e:
                    print(f"  Error iterating segments: {str(e)}")
                    continue
                
                print(f"  Contained by elements: {element_contained}")
                print(f"  Contained by tasks: {task_contained}")
                print(f"  Uncontained: {uncontained}")
                
                # Print sample segments
                if len(segments) > 0:
                    print("\n  Sample segments:")
                    try:
                        for i in range(min(2, len(segments))):
                            try:
                                segment = segments[i]
                                segment_id = decode_if_bytes(segment['segment_id'])
                                element_id = decode_if_bytes(segment['containing_element_id'])
                                task_id = decode_if_bytes(segment['containing_task_id'])
                                
                                print(f"    Segment {i+1}: {segment_id}")
                                print(f"      Type: {segment_type}")
                                print(f"      Element: {element_id or 'None'}")
                                print(f"      Task: {task_id or 'None'}")
                                
                                # Print duration information
                                if 'duration' in segment.dtype.names:
                                    try:
                                        print(f"      Duration: {segment['duration']:.2f}s")
                                    except Exception:
                                        print("      Duration: <Error>")
                                        
                                if 'start_time' in segment.dtype.names and 'end_time' in segment.dtype.names:
                                    try:
                                        print(f"      Time range: {segment['start_time']:.2f} - {segment['end_time']:.2f}")
                                    except Exception:
                                        print("      Time range: <Error>")
                            except Exception as e:
                                print(f"    Error processing segment {i+1}: {str(e)}")
                    except Exception as e:
                        print(f"  Error printing segment details: {str(e)}")
            except Exception as e:
                print(f"Error processing {segment_type} segments: {str(e)}")
    except Exception as e:
        print(f"Error accessing segment types: {str(e)}")

def verify_associations(h5f):
    """Verify task-element associations."""
    print_header("ASSOCIATION VERIFICATION")
    
    if 'tasks' not in h5f or 'table' not in h5f['tasks']:
        print("No tasks table found in file")
        return
        
    if 'elements' not in h5f or 'table' not in h5f['elements']:
        print("No elements table found in file")
        return
    
    # Safely get datasets
    tasks = safely_get_dataset(h5f, 'tasks/table')
    elements = safely_get_dataset(h5f, 'elements/table')
    
    if tasks is None or elements is None:
        print("Could not read tasks or elements table")
        return
    
    try:
        # Create a dictionary of task_id -> element_ids
        task_elements = {}
        for task in tasks:
            try:
                task_id = decode_if_bytes(task['task_id'])
                
                # Handle case where element_ids field might not exist or is empty
                if 'element_ids' in task.dtype.names:
                    try:
                        # Check if the field has data
                        if hasattr(task['element_ids'], 'size') and task['element_ids'].size > 0:
                            element_ids = [decode_if_bytes(eid) for eid in task['element_ids']]
                            task_elements[task_id] = set(element_ids)
                        else:
                            task_elements[task_id] = set()
                    except Exception as e:
                        print(f"  Warning: Could not read element_ids for task {task_id}: {str(e)}")
                        task_elements[task_id] = set()
                else:
                    task_elements[task_id] = set()
            except Exception as e:
                print(f"  Error processing task associations: {str(e)}")
                continue
        
        # Check if all elements have a valid task
        orphaned_elements = []
        for element in elements:
            try:
                element_id = decode_if_bytes(element['element_id'])
                task_id = decode_if_bytes(element['task_id'])
                
                if not task_id or task_id not in task_elements:
                    orphaned_elements.append((element_id, task_id))
            except Exception as e:
                print(f"  Error checking element task association: {str(e)}")
                continue
        
        if orphaned_elements:
            print(f"WARNING: Found {len(orphaned_elements)} elements without a valid task:")
            for i, (element_id, task_id) in enumerate(orphaned_elements[:10]):
                print(f"  Element '{element_id}' refers to task '{task_id}' which doesn't exist")
            if len(orphaned_elements) > 10:
                print(f"  ... and {len(orphaned_elements) - 10} more")
        else:
            print("All elements have valid task references")
        
        # Check if task element lists match element task references
        try:
            mismatch_count = 0
            for element in elements:
                try:
                    element_id = decode_if_bytes(element['element_id'])
                    task_id = decode_if_bytes(element['task_id'])
                    
                    if task_id in task_elements and element_id not in task_elements[task_id]:
                        mismatch_count += 1
                except Exception:
                    continue
            
            if mismatch_count:
                print(f"\nWARNING: Found {mismatch_count} elements not in their task's element list")
            else:
                print("All element-task associations are bidirectional")
        except Exception as e:
            print(f"Error verifying bidirectional associations: {str(e)}")
    except Exception as e:
        print(f"Error during association verification: {str(e)}")

def inspect_indices(h5f):
    """Inspect the index structures."""
    print_header("INDICES")
    
    if 'indices' not in h5f:
        print("No indices found in file")
        return
    
    try:
        indices = h5f['indices']
        print(f"Found {len(indices)} index types: {', '.join(indices.keys())}")
        
        # Print summary of each index type
        for index_type in indices:
            try:
                index_group = indices[index_type]
                print(f"\n{index_type}: {len(index_group)} entries")
                
                # Sample a few entries
                sample_keys = list(index_group.keys())[:3]
                if sample_keys:
                    print("  Sample entries:")
                    for key in sample_keys:
                        try:
                            dataset = index_group[key]
                            if dataset is None:
                                print(f"    {key}: <Could not read dataset>")
                                continue
                                
                            # Check if we can get the length
                            try:
                                entry_count = len(dataset)
                                print(f"    {key}: {entry_count} items")
                            except Exception:
                                print(f"    {key}: <Could not determine size>")
                                
                            # Try to read first few items if small enough
                            try:
                                if len(dataset) > 0 and len(dataset) <= 5:
                                    items = []
                                    for item in dataset[:]:
                                        items.append(decode_if_bytes(item))
                                    print(f"      Items: {items}")
                            except Exception:
                                pass
                        except Exception as e:
                            print(f"    {key}: <Error: {str(e)}>")
            except Exception as e:
                print(f"  Error processing index type {index_type}: {str(e)}")
    except Exception as e:
        print(f"Error accessing indices: {str(e)}")

def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: python inspect-event.py <h5_file_path>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    try:
        with h5py.File(file_path, 'r') as h5f:
            print(f"Inspecting: {file_path}")
            
            # Print file structure first
            print_header("FILE STRUCTURE")
            print_file_structure(h5f)
            
            # Now do detailed inspection
            inspect_metadata(h5f)
            inspect_tasks(h5f)
            inspect_elements(h5f)
            inspect_segments(h5f)
            verify_associations(h5f)
            inspect_indices(h5f)
            
            print("\nInspection completed successfully")
    except Exception as e:
        print(f"Error inspecting file: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()