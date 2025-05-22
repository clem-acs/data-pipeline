#!/usr/bin/env python
"""
Utility to inspect Zarr event files produced by t2C_event_v0.py transform.
Shows the structure and content of hierarchical Zarr stores.
Compatible with both Zarr v2 and v3 formats.
"""

import sys
import os
import argparse
import numpy as np
from collections import defaultdict
import warnings

# Suppress zarr warnings about vlen-utf8 codecs
warnings.filterwarnings("ignore", 
                       message="The codec `vlen-utf8` is currently not part in the Zarr format 3 specification")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Inspect Zarr event files")
    parser.add_argument("zarr_path", help="Path to Zarr store")
    parser.add_argument("--summary", "-s", action="store_true", help="Print summary only")
    parser.add_argument("--full", "-f", action="store_true", help="Print full details")
    parser.add_argument("--zarr-only", action="store_true", help="Use zarr API directly, bypass xarray")
    args = parser.parse_args()

    # Check if path exists
    if not os.path.exists(args.zarr_path):
        print(f"ERROR: Path does not exist: {args.zarr_path}")
        return 1

    try:
        import zarr
    except ImportError:
        print("ERROR: zarr module not installed. Please install it with:")
        print("  pip install zarr")
        return 1
    
    try:
        import s3fs
    except ImportError:
        print("WARNING: s3fs module not installed. S3 access will not be available.")
    
    try:
        import xarray as xr
    except ImportError:
        print("WARNING: xarray module not installed. Falling back to zarr-only mode.")
        args.zarr_only = True

    # Open the zarr store with multiple fallbacks
    root = None
    try:
        print(f"Opening Zarr store: {args.zarr_path}")
        # Simple direct open usually works with most zarr versions
        root = zarr.open(args.zarr_path, mode="r")
    except Exception as e:
        print(f"First open attempt failed: {e}")
        try:
            # Try with explicit store for zarr v2
            from zarr.storage import DirectoryStore
            store = DirectoryStore(args.zarr_path)
            root = zarr.open(store, mode="r")
        except Exception as e2:
            print(f"Second open attempt failed: {e2}")
            print("ERROR: All attempts to open the zarr store failed.")
            return 1

    # Determine Zarr format version (v2 or v3)
    zarr_version = 3  # Default to v3
    try:
        # Check for v2-specific attribute
        if hasattr(root, 'store') and hasattr(root.store, 'path'):
            if '.zgroup' in os.listdir(os.path.join(root.store.path)):
                zarr_version = 2
        # Another check for v2
        try:
            next(iter(root.store.values())).compressor  # Will raise exception for v3
            zarr_version = 2
        except (StopIteration, AttributeError, TypeError):
            pass
    except Exception:
        # If any errors, just assume v3
        pass
    
    print(f"Zarr Format Version: {zarr_version}")

    # Process with zarr API directly if requested or if xarray is not available
    if args.zarr_only:
        print("\n=== ZARR STORE STRUCTURE ===")
        process_store_directly(root, args)
        return 0

    # Try to open with xarray (with fallback for zarr v3)
    try:
        # Use zarr_path directly instead of the already opened root
        # This avoids xarray trying to access v3-specific attributes
        ds = xr.open_zarr(args.zarr_path)
        print("\n=== XARRAY DATASET SUMMARY ===")
        print(ds)
        
        # Check for hierarchical structure based on variable names
        if not args.summary:
            print("\n=== HIERARCHICAL GROUPS ===")
            # Check for variable patterns indicating groups
            elements_vars = [v for v in ds.data_vars if v.startswith("elements/")]
            tasks_vars = [v for v in ds.data_vars if v.startswith("tasks/")]
            segments_vars = [v for v in ds.data_vars if v.startswith("segments/")]
            
            # Check for old-style flat naming
            flat_elements = [v for v in ds.data_vars if v.startswith("element_")]
            flat_tasks = [v for v in ds.data_vars if v.startswith("task_")]
            flat_segments = [v for v in ds.data_vars if v.startswith("segment_")]
            
            # Analyze hierarchical structure if present
            if elements_vars or tasks_vars or segments_vars:
                print("Found hierarchical group structure with nested paths")
                
                groups = []
                if elements_vars:
                    groups.append(("elements", elements_vars))
                if tasks_vars:
                    groups.append(("tasks", tasks_vars))
                if segments_vars:
                    groups.append(("segments", segments_vars))
                
                for group_name, vars_list in groups:
                    print(f"\n--- {group_name.upper()} ---")
                    print(f"Variables: {len(vars_list)}")
                    
                    # Try to find dimension for this group
                    group_id_dim = f"{group_name[:-1]}_id"  # elements -> element_id
                    if group_id_dim in ds.dims:
                        print(f"Count: {len(ds.dims[group_id_dim])}")
                    
                    # Print variables details for full mode
                    if args.full:
                        for var in sorted(vars_list):
                            # Extract attribute name after group prefix
                            attr_name = var.split("/")[1]
                            var_shape = ds[var].shape
                            var_dtype = ds[var].dtype
                            print(f"  {attr_name}: shape={var_shape}, dtype={var_dtype}")
                            
                            # Try to print a sample value
                            if len(ds[var]) > 0:
                                try:
                                    sample = ds[var].values[0]
                                    if isinstance(sample, (str, int, float, bool, np.number)):
                                        print(f"    Sample: {sample}")
                                except:
                                    pass
            
            # Handle flat structure (old version) if present and no hierarchical found
            elif flat_elements or flat_tasks or flat_segments:
                print("Found flat structure with prefixed variable names")
                
                if flat_elements:
                    print(f"\n--- ELEMENTS ---")
                    print(f"Variables: {len(flat_elements)}")
                    if 'element_id' in ds.dims:
                        print(f"Count: {len(ds.dims['element_id'])}")
                    if args.full:
                        for var in sorted(flat_elements):
                            print(f"  {var}: shape={ds[var].shape}, dtype={ds[var].dtype}")
                
                if flat_segments:
                    print(f"\n--- SEGMENTS ---")
                    print(f"Variables: {len(flat_segments)}")
                    if 'segment_id' in ds.dims:
                        print(f"Count: {len(ds.dims['segment_id'])}")
                    if args.full:
                        for var in sorted(flat_segments):
                            print(f"  {var}: shape={ds[var].shape}, dtype={ds[var].dtype}")
                
                if flat_tasks:
                    print(f"\n--- TASKS ---")
                    print(f"Variables: {len(flat_tasks)}")
                    if 'task_id' in ds.dims:
                        print(f"Count: {len(ds.dims['task_id'])}")
                    if args.full:
                        for var in sorted(flat_tasks):
                            print(f"  {var}: shape={ds[var].shape}, dtype={ds[var].dtype}")
            else:
                print("No recognized group structure found.")
            
            # Print dataset attributes
            if ds.attrs:
                print("\n=== ATTRIBUTES ===")
                for name, value in ds.attrs.items():
                    print(f"{name}: {value}")
    
    except Exception as e:
        print(f"Error opening with xarray: {str(e)}")
        print("This is expected with Zarr v3 stores. Falling back to direct zarr inspection.")
        process_store_directly(root, args)
    
    return 0

def process_store_directly(root, args):
    """Process the zarr store directly using the zarr API."""
    try:
        import zarr
        print("\n=== ZARR STRUCTURE ===")
        # Handle different zarr versions
        if hasattr(root, 'tree') and callable(root.tree):
            try:
                # This will work for zarr v2
                tree_output = root.tree()
                # For Windows cmd, handle unicode safely
                try:
                    print(tree_output)
                except UnicodeEncodeError:
                    # Fallback for Windows console encoding issues
                    print(tree_output.encode('ascii', 'replace').decode())
            except Exception as e:
                print(f"Error printing tree: {e}")
                print_hierarchical_structure(root)
        else:
            # For zarr v3 or missing tree() method
            print_hierarchical_structure(root)
        
        # Check if hierarchical or flat structure
        # First, check for actual hierarchical groups (zarr groups)
        group_names = []
        array_names = []
        for name in root:
            try:
                if isinstance(root[name], zarr.Group):
                    group_names.append(name)
                else:
                    array_names.append(name)
            except Exception:
                # If can't determine type, try to check differently
                try:
                    # Check if it has items method (a sign it's a group)
                    if hasattr(root[name], 'items'):
                        group_names.append(name)
                    else:
                        array_names.append(name)
                except:
                    # If all else fails, assume it's an array
                    array_names.append(name)
        
        # Process based on identified structure
        if 'elements' in group_names or 'tasks' in group_names or 'segments' in group_names:
            # Hierarchical structure found
            print("\n=== HIERARCHICAL GROUPS FOUND ===")
            
            for group_name in ['elements', 'tasks', 'segments']:
                if group_name in group_names:
                    print(f"\n--- {group_name.upper()} ---")
                    try:
                        group = root[group_name]
                        # List arrays in this group
                        group_arrays = list_arrays_in_group(group)
                        print(f"Variables: {len(group_arrays)}")
                        
                        # Try to get count from first array
                        if group_arrays:
                            first_array = group[group_arrays[0]]
                            count = first_array.shape[0] if first_array.shape else 0
                            print(f"Count: {count}")
                        
                        # Print details in full mode
                        if not args.summary and group_arrays:
                            print("Variables:")
                            for array_name in sorted(group_arrays):
                                try:
                                    array = group[array_name]
                                    print(f"  {array_name}: shape={array.shape}, dtype={array.dtype}")
                                    
                                    # Print sample for full mode
                                    if args.full and array.shape and array.shape[0] > 0:
                                        try:
                                            sample = array[0]
                                            if isinstance(sample, (str, int, float, bool, np.number)):
                                                print(f"    Sample: {sample}")
                                        except:
                                            pass
                                except Exception as e:
                                    print(f"  {array_name}: ERROR - {str(e)}")
                    except Exception as e:
                        print(f"Error accessing group {group_name}: {str(e)}")
        else:
            # Check for flat structure with prefixed arrays
            element_arrays = [name for name in array_names if name.startswith("element_")]
            task_arrays = [name for name in array_names if name.startswith("task_")]
            segment_arrays = [name for name in array_names if name.startswith("segment_")]
            
            if element_arrays or task_arrays or segment_arrays:
                print("\n=== FLAT STRUCTURE FOUND ===")
                
                if element_arrays:
                    print(f"\n--- ELEMENTS ---")
                    print(f"Variables: {len(element_arrays)}")
                    if element_arrays:
                        try:
                            first_array = root[element_arrays[0]]
                            print(f"Count: {first_array.shape[0] if first_array.shape else 0}")
                            
                            if not args.summary:
                                print("Variables:")
                                for array_name in sorted(element_arrays):
                                    array = root[array_name]
                                    print(f"  {array_name}: shape={array.shape}, dtype={array.dtype}")
                        except Exception as e:
                            print(f"Error accessing element arrays: {str(e)}")
                
                if segment_arrays:
                    print(f"\n--- SEGMENTS ---")
                    print(f"Variables: {len(segment_arrays)}")
                    if segment_arrays:
                        try:
                            first_array = root[segment_arrays[0]]
                            print(f"Count: {first_array.shape[0] if first_array.shape else 0}")
                            
                            if not args.summary:
                                print("Variables:")
                                for array_name in sorted(segment_arrays):
                                    array = root[array_name]
                                    print(f"  {array_name}: shape={array.shape}, dtype={array.dtype}")
                        except Exception as e:
                            print(f"Error accessing segment arrays: {str(e)}")
                
                if task_arrays:
                    print(f"\n--- TASKS ---")
                    print(f"Variables: {len(task_arrays)}")
                    if task_arrays:
                        try:
                            first_array = root[task_arrays[0]]
                            print(f"Count: {first_array.shape[0] if first_array.shape else 0}")
                            
                            if not args.summary:
                                print("Variables:")
                                for array_name in sorted(task_arrays):
                                    array = root[array_name]
                                    print(f"  {array_name}: shape={array.shape}, dtype={array.dtype}")
                        except Exception as e:
                            print(f"Error accessing task arrays: {str(e)}")
            else:
                print("\n=== UNKNOWN STRUCTURE ===")
                print(f"Found {len(array_names)} arrays at root level.")
                
                if not args.summary and array_names:
                    print("Top-level arrays:")
                    for array_name in sorted(array_names[:10]):  # Show first 10
                        try:
                            array = root[array_name]
                            print(f"  {array_name}: shape={array.shape}, dtype={array.dtype}")
                        except Exception as e:
                            print(f"  {array_name}: ERROR - {str(e)}")
                    
                    if len(array_names) > 10:
                        print(f"  ... and {len(array_names)-10} more arrays")
        
        # Print attributes if available
        try:
            if hasattr(root, 'attrs'):
                attrs = {}
                try:
                    # For zarr v2
                    if hasattr(root.attrs, 'asdict'):
                        attrs = root.attrs.asdict()
                    # For zarr v3
                    else:
                        attrs = dict(root.attrs)
                except:
                    pass
                
                if attrs and not args.summary:
                    print("\n=== ATTRIBUTES ===")
                    for name, value in attrs.items():
                        print(f"{name}: {value}")
        except Exception as e:
            print(f"Error accessing attributes: {str(e)}")
    
    except Exception as e:
        print(f"Error in direct zarr processing: {str(e)}")

def print_hierarchical_structure(group, prefix=""):
    """Print a hierarchical structure of a zarr group."""
    try:
        import zarr
        for name in group:
            try:
                item = group[name]
                # Check if it's a group or array
                is_group = False
                try:
                    is_group = isinstance(item, zarr.Group)
                except:
                    # Fallback check
                    is_group = hasattr(item, 'items')
                
                if is_group:
                    print(f"{prefix}{name}/ (Group)")
                    print_hierarchical_structure(item, prefix + "  ")
                else:
                    shape_str = str(item.shape) if hasattr(item, 'shape') else "?"
                    dtype_str = str(item.dtype) if hasattr(item, 'dtype') else "?"
                    print(f"{prefix}{name}: shape={shape_str}, dtype={dtype_str}")
            except Exception as e:
                print(f"{prefix}{name}: ERROR - {str(e)}")
    except Exception as e:
        print(f"Error in print_hierarchical_structure: {str(e)}")

def list_arrays_in_group(group):
    """List all arrays in a zarr group."""
    arrays = []
    try:
        import zarr
        for name in group:
            try:
                item = group[name]
                # Try various ways to determine if it's an array
                is_array = False
                try:
                    # Most reliable check
                    is_array = not isinstance(item, zarr.Group)
                except:
                    # Fallback check - arrays have shape, groups don't
                    is_array = hasattr(item, 'shape') and not hasattr(item, 'items')
                
                if is_array:
                    arrays.append(name)
            except:
                pass  # Skip items we can't classify
    except Exception:
        pass  # Return what we've found so far
    
    return arrays

if __name__ == "__main__":
    sys.exit(main())