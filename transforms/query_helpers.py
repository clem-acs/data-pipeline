"""
Query Helper Functions

Utilities for working with the QueryTransform, providing common operations
for filtering elements, handling data types, and collecting windows.
"""

from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
import zarr


def bytesify(np_arr: np.ndarray) -> np.ndarray:
    """
    Convert any object/unicode array to fixed-width UTF-8 bytes.
    
    Zarr v3 forbids object dtype; this function determines the appropriate
    string length and converts to a fixed-width bytes dtype.
    
    Args:
        np_arr: NumPy array potentially containing object/unicode strings
        
    Returns:
        NumPy array with bytes dtype
    """
    # Only process object or unicode dtypes
    if np_arr.dtype.kind not in ("O", "U"):
        return np_arr
    
    # Convert to strings
    as_str = np_arr.astype(str)
    
    # Find the maximum string length in the array
    max_len = max(1, max(len(s) for s in as_str))
    
    # Convert to fixed-width bytes dtype with maximum length
    return as_str.astype(f"S{max_len}")


def sanitize_session_id(session_id: str) -> str:
    """
    Prevent '/' from being interpreted as nested subgroups in Zarr.
    
    Args:
        session_id: Original session ID
        
    Returns:
        Sanitized session ID
    """
    return session_id.replace("/", "%2F")


def filter_elements(elements: Dict[str, np.ndarray], 
                    **predicates) -> Optional[Dict[str, np.ndarray]]:
    """
    Generic boolean mask filter on the elements dict.
    
    Example:
        eye_elements = filter_elements(elements, task_type="eye", input_modality="visual")
    
    Args:
        elements: Dictionary of element data arrays
        **predicates: Key-value pairs to filter on (e.g., task_type="eye")
        
    Returns:
        Filtered dictionary or None if no matches
    """
    if not elements:
        return None
    
    # Start with all elements included
    mask = np.ones(elements["element_ids"].shape[0], dtype=bool)
    
    # Apply each predicate
    for col, wanted in predicates.items():
        if col not in elements:
            return None
        
        col_data = elements[col]
        if col_data.dtype.kind in ("S", "a"):  # Handle byte strings
            col_data = col_data.astype(str)
        
        # Update mask to match predicate
        mask &= (col_data == wanted)
    
    # Return None if no matches
    if not np.any(mask):
        return None
    
    # Apply mask to each array and return
    return {
        k: v[mask] if isinstance(v, np.ndarray) and v.shape[0] == mask.shape[0] else v 
        for k, v in elements.items()
    }


def join_elements_tasks(elements: Dict[str, np.ndarray], 
                         tasks: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Join element and task data based on task_id.
    
    Args:
        elements: Dictionary of element arrays
        tasks: Dictionary of task arrays
        
    Returns:
        Combined dictionary with elements + associated task data
    """
    if "task_id" not in elements:
        return elements
    
    # Create a copy of the elements dictionary
    result = elements.copy()
    
    # Convert task_id to string for comparison if needed
    element_task_ids = elements["task_id"]
    if element_task_ids.dtype.kind in ("S", "a"):
        element_task_ids = element_task_ids.astype(str)
    
    # Extract task IDs from tasks dictionary
    if "task_id" not in tasks:
        return elements
    
    task_ids = tasks["task_id"]
    if task_ids.dtype.kind in ("S", "a"):
        task_ids = task_ids.astype(str)
    
    # For each task variable, create corresponding element-mapped array
    for task_var in tasks:
        if task_var == "task_id" or task_var in result:
            continue
            
        # Create array to hold task data for each element
        task_data = np.zeros(elements["element_ids"].shape[0], dtype=tasks[task_var].dtype)
        
        # Map task data to elements
        for i, element_task_id in enumerate(element_task_ids):
            # Find matching task
            matches = np.where(task_ids == element_task_id)[0]
            if matches.size > 0:
                # Use first match if multiple
                task_data[i] = tasks[task_var][matches[0]]
        
        # Add to result
        result[f"task_{task_var}"] = task_data
    
    return result


def extract_full_windows(
    wgroup: zarr.Group,
    hits: np.ndarray          # indices of the windows you want
) -> Dict[str, np.ndarray]:
    """
    Return every variable whose first dimension matches the time dimension.
    
    This efficiently slices all window data arrays with the same indices.
    
    Args:
        wgroup: Zarr group containing window data
        hits: Array of window indices to extract
        
    Returns:
        Dictionary with all arrays that have a matching first dimension
    """
    # Add debug print
    import logging
    logging.debug(f"extract_full_windows called with {len(hits)} indices: min={hits.min() if len(hits) > 0 else -1}, max={hits.max() if len(hits) > 0 else -1}")
    
    out = {}
    time_arr_length = wgroup["time"].shape[0]
    logging.debug(f"Time array length in wgroup: {time_arr_length}")
    
    # Add debug print here
    import logging
    for key in wgroup.array_keys():
        arr = wgroup[key]
        logging.debug(f"Array {key} shape={arr.shape}, matches time?={arr.shape[0] == time_arr_length if arr.shape else False}")
    
    # Debug prints about the hits array BEFORE processing any arrays
    # Use print directly to ensure it shows up in output
    print(f"BEFORE PROCESSING: hits len={len(hits)}, unique={len(np.unique(hits))}, "
          f"sorted_ok={(hits[:-1] < hits[1:]).all() if len(hits) > 1 else True}, "
          f"max_hit={hits.max() if len(hits) > 0 else -1}, time_len={wgroup['time'].shape[0]}")
    print(f"BEFORE PROCESSING: unique={len(np.unique(hits))}  total={len(hits)}  "
          f"sorted={(hits[:-1] < hits[1:]).all() if len(hits) > 1 else True}")
    
    for key in wgroup.array_keys():
        arr = wgroup[key]
        if arr.shape and arr.shape[0] == time_arr_length:
            # Use vindex for efficient chunk-wise slicing
            import logging
            logging.debug(
                f"Processing {key}: array shape={arr.shape[0]}, "
                f"hit indices range {hits.min() if len(hits) > 0 else -1}-{hits.max() if len(hits) > 0 else -1}"
            )
            
            # Check for invalid indices before using vindex
            invalid_indices = hits[hits >= arr.shape[0]]
            if len(invalid_indices) > 0:
                logging.warning(f"Found {len(invalid_indices)} out-of-bounds indices: {invalid_indices[:5]}{'...' if len(invalid_indices) > 5 else ''}")
                valid_hits = hits[hits < arr.shape[0]]
                if len(valid_hits) == 0:
                    logging.warning(f"No valid indices for {key}, skipping")
                    continue
                hits_to_use = valid_hits
            else:
                hits_to_use = hits
            
            # Explicitly build selection tuple with slices for remaining dimensions
            # Avoid using ellipsis completely as it causes selection.index(Ellipsis) to fail with NumPy arrays
            sel = (hits_to_use,) + tuple(slice(None) for _ in range(arr.ndim - 1))
            out[key] = arr.oindex[sel]
    
    import logging
    logging.info(f"Extracted {len(hits)} windows with {len(out)} matching arrays")
    return out


def build_labels(elems: Dict[str, np.ndarray], hits: np.ndarray, times: np.ndarray, 
                label_map: Optional[Dict[str, Union[str, int]]] = None) -> np.ndarray:
    """
    Build labels array for window indices based on which element they fall into.
    
    Args:
        elems: Dictionary with element data (element_ids, start_time, end_time)
        hits: Array of window indices that match
        times: Array of timestamps for each window
        label_map: Optional mapping from patterns to label values
        
    Returns:
        Array of labels for each hit
    """
    if label_map is None:
        # Return numeric labels directly for more efficient processing
        label_map = {"closed": 0, "open": 1, "intro": 2, "unknown": 3}
    
    labels = np.full(len(hits), label_map.get("unknown", 3), dtype=np.int32)
    
    for i, hit_idx in enumerate(hits):
        hit_time = times[hit_idx]
        # Find the element this hit belongs to
        for j, (eid, start, end) in enumerate(zip(elems["element_ids"], 
                                                elems["start_time"], 
                                                elems["end_time"])):
            if start <= hit_time <= end:
                # Convert element_id to string for pattern matching
                eid_str = eid.decode("utf-8") if isinstance(eid, (bytes, np.bytes_)) else str(eid)
                
                # Determine label from element_id patterns
                for pattern, value in label_map.items():
                    if pattern in eid_str:
                        labels[i] = value
                        break
                
                break
    
    return labels


def build_eids(elems: Dict[str, np.ndarray], hits: np.ndarray, times: np.ndarray) -> np.ndarray:
    """
    Build element_ids array for window indices based on which element they fall into.
    
    Args:
        elems: Dictionary with element data (element_ids, start_time, end_time)
        hits: Array of window indices that match
        times: Array of timestamps for each window
        
    Returns:
        Array of element_ids for each hit
    """
    eids = np.empty(len(hits), dtype=object)
    
    for i, hit_idx in enumerate(hits):
        hit_time = times[hit_idx]
        # Find the element this hit belongs to
        for eid, start, end in zip(elems["element_ids"], 
                                 elems["start_time"], 
                                 elems["end_time"]):
            if start <= hit_time <= end:
                eids[i] = eid
                break
    
    return eids


def collect_windows(
    window_group: zarr.Group,
    time_arr: np.ndarray,
    intervals: List[Tuple[Any, Tuple[float, float]]],
    data_var: str,
    label_map: Optional[Dict[str, Any]] = None,
    cap: int = 100_000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pull windows whose timestamp lies in any of the (start, end) intervals.
    
    Args:
        window_group: Zarr group containing window data
        time_arr: Array of timestamps for each window
        intervals: List of (element_id, (start_time, end_time)) tuples
        data_var: Name of the variable containing the window data
        label_map: Optional dict mapping element_id patterns to labels
        cap: Maximum number of windows to extract
        
    Returns:
        Tuple of (windows, labels, element_ids) arrays
    """
    windows = []
    labels = []
    elem_ids = []
    
    # Default label mapping if none provided
    if label_map is None:
        label_map = {
            "closed": "closed",
            "open": "open", 
            "intro": "intro"
        }
    
    for element_id, (start, end) in intervals:
        # Find windows where time falls within the interval
        hits = np.where((time_arr >= start) & (time_arr <= end))[0]
        if not hits.size:
            continue
        
        for idx in hits:
            if len(windows) >= cap:
                return (np.asarray(windows), 
                        np.asarray(labels), 
                        np.asarray(elem_ids))
            
            # Extract window data
            win = window_group[data_var][idx]
            windows.append(win)
            
            # Handle element_id type conversion
            eid_str = element_id
            if isinstance(eid_str, (bytes, np.bytes_)):
                eid_str = eid_str.decode("utf-8")
            eid_str = str(eid_str)
            elem_ids.append(eid_str)
            
            # Determine label based on element_id content
            label = "unknown"
            for pattern, label_value in label_map.items():
                if pattern in eid_str:
                    label = label_value
                    break
            labels.append(label)
    
    return (np.asarray(windows), 
            np.asarray(labels), 
            np.asarray(elem_ids))


def get_neural_windows_dataset(sessions_group: zarr.Group, 
                               session_ids: List[str],
                               label_map: Optional[Dict[str, int]] = None):
    """
    Extract neural windows data from all sessions and combine into one dataset.
    
    Args:
        sessions_group: Zarr group containing session subgroups
        session_ids: List of session IDs to include
        label_map: Optional mapping from string labels to numeric indices
        
    Returns:
        torch.utils.data.TensorDataset with windows and labels
    """
    import torch
    import numpy as np
    
    # Default label mapping if none provided
    if label_map is None:
        label_map = {"closed": 0, "open": 1, "intro": 2, "unknown": 3}
    
    # Collect windows and labels from all sessions
    all_windows = []
    all_labels = []
    window_count = 0
    
    # Process each session
    for session_id in session_ids:
        session_group = sessions_group[sanitize_session_id(session_id)]
        
        # Check if this session has window data
        if "windows" not in session_group or "labels" not in session_group:
            continue
            
        # Read windows and labels
        windows = session_group["windows"][:]
        labels = session_group["labels"][:]
        
        # Convert labels to strings if they're bytes
        if labels.dtype.kind in ("S", "a"):
            labels = labels.astype(str)
        
        # Append to our collections
        all_windows.append(windows)
        all_labels.append(labels)
        window_count += len(windows)
    
    if not all_windows:
        return None
    
    # Concatenate data from all sessions
    windows_array = np.concatenate(all_windows, axis=0)
    labels_array = np.concatenate(all_labels, axis=0)
    
    # Convert string labels to numeric indices
    numeric_labels = np.array([label_map.get(label, label_map.get("unknown", 3)) 
                               for label in labels_array])
    
    # Convert to torch tensors
    windows_tensor = torch.tensor(windows_array, dtype=torch.float32)
    labels_tensor = torch.tensor(numeric_labels, dtype=torch.long)
    
    # Create the dataset
    return torch.utils.data.TensorDataset(windows_tensor, labels_tensor)