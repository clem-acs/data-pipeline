"""
Query Helper Functions

Utilities for working with the QueryTransform, providing common operations
for filtering elements, handling data types, and collecting windows.
"""

from typing import Dict, List, Tuple, Any, Optional, Union, Callable
import numpy as np
import zarr
import numcodecs
import logging
import time

# No default compression - let Zarr handle compression automatically


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
    
    # Convert to strings and handle various Python types
    str_array = np.array([str(x) for x in np_arr.flatten()]).reshape(np_arr.shape)
    
    # Find the maximum string length in bytes after UTF-8 encoding
    max_len = max(1, max(len(s.encode('utf-8')) for s in str_array.flatten()))
    
    # Create an array of the appropriate size for UTF-8 encoded strings
    return np.array([s.encode('utf-8') for s in str_array.flatten()], 
                   dtype=f'S{max_len}').reshape(np_arr.shape)


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




def extract_full_windows(
    wgroup: zarr.Group,
    hits: np.ndarray,          # indices of the windows you want
    logger: Optional[logging.Logger] = None
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
    log_debug = logger.debug if logger else logging.debug
    log_debug(f"extract_full_windows called with {len(hits)} indices: min={hits.min() if len(hits) > 0 else -1}, max={hits.max() if len(hits) > 0 else -1}")
    
    out = {}
    time_arr_length = wgroup["time"].shape[0]
    log_debug(f"Time array length in wgroup: {time_arr_length}")
    
    # Add debug print here
    for key in wgroup.array_keys():
        arr = wgroup[key]
        log_debug(f"Array {key} shape={arr.shape}, matches time?={arr.shape[0] == time_arr_length if arr.shape else False}")
    
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
            log_debug(
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
    
    log_info = logger.info if logger else logging.info
    log_info(f"Extracted {len(hits)} windows with {len(out)} matching arrays")
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
        # Generic fallback with single unknown label
        label_map = {"unknown": 100}
    
    labels = np.full(len(hits), label_map.get("unknown", 100), dtype=np.int32)
    
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




def window_indices_for_range(time_arr: np.ndarray, start: float, end: float) -> np.ndarray:
    """
    Find indices of windows whose timestamps fall within a given time range.
    
    Args:
        time_arr: Array of timestamps for each window
        start: Start timestamp
        end: End timestamp
        
    Returns:
        Array of indices matching the time range
    """
    return np.where((time_arr >= start) & (time_arr <= end))[0]


def find_element_for_timestamp(elements: Dict[str, np.ndarray], timestamp: float) -> Optional[Any]:
    """
    Find which element contains a specific timestamp.
    
    Args:
        elements: Dictionary with element data (element_ids, start_time, end_time)
        timestamp: Timestamp to find
        
    Returns:
        Element ID containing the timestamp or None
    """
    for eid, start, end in zip(elements["element_ids"], elements["start_time"], elements["end_time"]):
        if start <= timestamp <= end:
            return eid
    return None


def extract_lang_tokens(zgroup: zarr.Group, group_name: str) -> Optional[Dict[str, Any]]:
    """
    Extract token data from a language zarr group.
    
    Args:
        zgroup: Zarr group containing language data
        group_name: Language group name (L, W, R, S, W_corrected)
        
    Returns:
        Dictionary with token data or None if group doesn't exist
    """
    # Check if language group exists
    if 'language' not in zgroup:
        return None
        
    # Check if specific language subgroup exists
    if group_name not in zgroup['language']:
        return None
        
    group = zgroup['language'][group_name]
    result = {}
    
    # Extract text
    if 'text' in group:
        result['text'] = group['text'][()]
    elif 'text' in group.attrs:
        result['text'] = group.attrs.get('text', '')
    else:
        # Default to empty text if not found
        result['text'] = ''
    
    # Extract token arrays
    tokens = []
    if 'token' not in group:
        return result  # Return just the text if no tokens
        
    token_count = group['token'].shape[0]  # Use shape attribute instead of len()
    
    for i in range(token_count):
        token_data = {}
        
        # Add common attributes
        for attr in ['token', 'token_id', 'start_timestamp', 'end_timestamp', 'special_token']:
            if attr in group:
                token_data[attr] = group[attr][i]
        
        # Add group-specific attributes
        if group_name == 'W':
            for attr in ['keystroke_events', 'trigger_keystrokes']:
                if attr in group:
                    token_data[attr] = group[attr][i]
            
            # Handle char_indices
            if 'char_indices' in group and len(group['char_indices']) > i:
                token_data['char_indices'] = list(group['char_indices'][i])
            elif 'char_indices_arrays' in group and str(i) in group['char_indices_arrays']:
                token_data['char_indices'] = list(group['char_indices_arrays'][str(i)][:])
            else:
                token_data['char_indices'] = []
                
        elif group_name == 'W_corrected':
            if 'unchanged' in group:
                token_data['unchanged'] = group['unchanged'][i]
                
            # Handle original_indices
            if 'original_indices' in group and len(group['original_indices']) > i:
                token_data['original_indices'] = list(group['original_indices'][i])
            elif 'original_indices_arrays' in group and str(i) in group['original_indices_arrays']:
                token_data['original_indices'] = list(group['original_indices_arrays'][str(i)][:])
            else:
                token_data['original_indices'] = []
        
        tokens.append(token_data)
    
    result['tokens'] = tokens
    
    # Add additional attributes for W_corrected
    if group_name == 'W_corrected':
        if 'correctness_score' in group:
            result['correctness_score'] = group['correctness_score'][()]
        elif 'correctness_score' in group.attrs:
            result['correctness_score'] = group.attrs.get('correctness_score', 0.0)
            
        if 'original_text' in group:
            result['original_text'] = group['original_text'][()]
        elif 'original_text' in group.attrs:
            result['original_text'] = group.attrs.get('original_text', '')
    
    return result




# =============================================================================
# Zarr Management Functions
# =============================================================================

def open_zarr_safely(uri: str, *, mode: str = "r", logger: Optional[logging.Logger] = None, storage_options: Optional[Dict[str, Any]] = None) -> zarr.Group:
    """
    Zarr-v3-only opener that guarantees fresh consolidated metadata.

    Steps:
    1. Try fast open (`use_consolidated=True`).
    2. If that fails *or* returns an apparently empty group,
       re-open without consolidated metadata,
       regenerate the consolidated block right away,
       then re-open again with `use_consolidated=True`.
    
    Args:
        uri: URI for the zarr store (e.g., s3://bucket/path/to/store.zarr)
        mode: Access mode (default: "r" for read-only)
        logger: Optional logger (uses print statements if None)
        
    Returns:
        Open zarr.Group object with fresh consolidated metadata
    """
    log = logger.info if logger else print
    debug = logger.debug if logger else print
    error = logger.error if logger else print
    
    log(f"Opening zarr store: {uri}")
    opts = storage_options or {"anon": False}

    # ----- 1 · fast path -------------------------------------------------
    try:
        debug(f"Trying fast path with use_consolidated=True...")
        g = zarr.open_group(
            store=uri,
            mode=mode,
            storage_options=opts,
            use_consolidated=True,      # <-- v3 keyword
        )
        arrays = list(g.array_keys())
        groups = list(g.group_keys())
        debug(f"Fast path - arrays: {arrays}, groups: {groups}")
        
        # If the metadata is valid, arrays or sub-groups will be visible.
        if arrays or groups:
            debug(f"Fast path SUCCESS - using consolidated metadata")
            return g
        # Otherwise the block is present but stale → fall through.
        debug(f"Fast path: consolidated metadata exists but empty, rebuilding...")
    except KeyError as e:
        # No consolidated block yet → fall through to rebuild.
        debug(f"Fast path FAILED with KeyError: {e}, rebuilding metadata...")
    except Exception as e:
        debug(f"Fast path FAILED with unexpected error: {type(e).__name__}:{e}, rebuilding metadata...")

    # ----- 2 · rebuild consolidated metadata ----------------------------
    debug(f"Opening in write mode to rebuild metadata...")
    # Open with a slow tree-walk so we can see everything.
    try:
        g = zarr.open_group(
            store=uri,
            mode="a",               # need write access to store new metadata
            storage_options=opts,
            use_consolidated=False, # explicit to be clear
        )
        arrays = list(g.array_keys())
        groups = list(g.group_keys())
        debug(f"Before consolidation - arrays: {arrays}, groups: {groups}")
        debug(f"Group directly contains 'elements'?: {'elements' in g}")
        
        if 'elements' in g:
            debug(f"Elements contents: {list(g['elements'].array_keys())}")

        # Write (or overwrite) the inline "consolidated_metadata" section
        # inside zarr.json.  For v3 this is a cheap local write; no full copy.
        debug(f"Consolidating metadata...")
        zarr.consolidate_metadata(g.store)
        debug(f"Metadata consolidated, reopening...")
    except Exception as e:
        error(f"ERROR during metadata rebuild: {type(e).__name__}:{e}")
        raise

    # Re-open in read mode with the fresh metadata block.
    try:
        debug(f"Final re-open with use_consolidated=True...")
        g_final = zarr.open_group(
            store=uri,
            mode=mode,
            storage_options=opts,
            use_consolidated=True,
        )
        arrays = list(g_final.array_keys())
        groups = list(g_final.group_keys())
        debug(f"Final result - arrays: {arrays}, groups: {groups}")
        debug(f"Final group directly contains 'elements'?: {'elements' in g_final}")
        
        if 'elements' in g_final:
            debug(f"Final elements contents: {list(g_final['elements'].array_keys())}")
            
        return g_final
    except Exception as e:
        error(f"ERROR during final re-open: {type(e).__name__}:{e}")
        raise


def capped_chunks(shape, cap=100):
    """
    Return a tuple such that each chunk size is min(cap, dim_size).
    
    Args:
        shape: Array shape (tuple or shape-like object)
        cap: Maximum size for any dimension in a chunk
        
    Returns:
        Tuple with same length as shape, with each value capped
    """
    return tuple(min(cap, d) for d in shape)


def save_obj_to_zarr(parent_group: zarr.Group, key: str, value: Any, 
                    logger: Optional[logging.Logger] = None, indent: str = ""):
    """
    Recursively save any Python object to a zarr hierarchy.
    
    Args:
        parent_group: Parent zarr group
        key: Name for this item
        value: Any Python object (dict, list, array, scalar)
        logger: Optional logger (uses print statements if None)
        indent: String for indentation in debug prints
    """
    log = logger.info if logger else print
    debug = logger.debug if logger else print
    warning = logger.warning if logger else print
    
    # Sanitize key for zarr path safety
    safe_key = str(key).replace("/", "_")
    
    debug(f"{indent}Processing: {safe_key}, type: {type(value)}")
    
    try:
        if isinstance(value, dict):
            # Dictionary → zarr group with named items
            debug(f"{indent}Creating group: {safe_key} with {len(value)} items")
            group = parent_group.create_group(safe_key, overwrite=True)
            debug(f"{indent}Group {safe_key} created successfully")
            
            # Process each item in the dictionary
            for k, v in value.items():
                save_obj_to_zarr(group, k, v, logger, indent + "  ")
                
        elif isinstance(value, list):
            # List → zarr group with numbered items
            debug(f"{indent}Creating list group: {safe_key} with {len(value)} items")
            group = parent_group.create_group(safe_key, overwrite=True)
            debug(f"{indent}List group {safe_key} created successfully")
            
            # Save each list item with its index as the key
            for i, item in enumerate(value):
                save_obj_to_zarr(group, str(i), item, logger, indent + "  ")
                
        elif isinstance(value, np.ndarray):
            # Array → zarr dataset
            if value.ndim == 0:
                # Store scalar arrays as attributes to avoid v3 0-D bug
                scalar_value = value.item()
                debug(f"{indent}Storing scalar array as attribute: {safe_key}={scalar_value}")
                parent_group.attrs[safe_key] = scalar_value
                debug(f"{indent}Scalar attribute {safe_key} created successfully")
                return  # Skip the rest of the processing
            else:
                # Regular multi-dimensional array handling
                value = bytesify(value)
                
                # Create properly dimensioned chunks for any array shape
                chunk_spec = capped_chunks(value.shape)
                
                # Build kwargs dictionary for create_dataset - no compression specified
                kwargs = dict(
                    name=safe_key,
                    shape=value.shape,
                    dtype=value.dtype,
                    data=value,
                    chunks=chunk_spec,
                    overwrite=True,
                )
                
                debug(f"{indent}Creating dataset: {safe_key}, shape={value.shape}, dtype={value.dtype}, chunks={chunk_spec}")
                debug(f"{indent}{safe_key}: dtype={value.dtype}, kind={value.dtype.kind}")
                
                parent_group.create_dataset(**kwargs)
                debug(f"{indent}Dataset {safe_key} created successfully")
            
        elif isinstance(value, (str, int, float, bool, type(None))):
            # Scalar → zarr attribute
            parent_group.attrs[safe_key] = value
            debug(f"{indent}Added scalar attribute: {safe_key}={value}")
            
        else:
            # Unsupported type → string representation as attribute
            str_value = str(value)
            parent_group.attrs[safe_key] = str_value
            debug(f"{indent}Converted to string attribute: {safe_key}={str_value[:30]}...")
            
    except Exception as e:
        debug(f"{indent}ERROR processing {safe_key}: {type(e).__name__}: {e}")
        warning(f"Error saving {safe_key}: {type(e).__name__}: {e}")


def save_session_to_subgroup(sessions_group: zarr.Group, session_id: str, 
                            data: Dict[str, Any], logger: Optional[logging.Logger] = None,
                            overwrite: bool = False):
    """
    Save session data to a session-specific subgroup.
    
    Args:
        sessions_group: Parent zarr group for all sessions
        session_id: Session ID being processed
        data: Dictionary with query results
        logger: Optional logger (uses print statements if None)
    """
    log = logger.info if logger else print
    debug = logger.debug if logger else print
    
    # Get the sanitized session key
    session_key = sanitize_session_id(session_id)
    
    debug(f"Saving session to subgroup: {session_key}")
    debug(f"Input data keys: {list(data.keys())}")
    
    # Create the session group
    try:
        sg = sessions_group.create_group(session_key, overwrite=overwrite)
        debug(f"Created session group: {session_key}")
    except Exception as e:
        debug(f"ERROR creating session group: {type(e).__name__}: {e}")
        raise
    
    # Handle session_id specially
    if "session_id" in data:
        sg.attrs["session_id"] = data["session_id"]
        debug(f"Added session_id to attrs: {data['session_id']}")
    
    # Save all other items using the recursive helper
    for key, val in data.items():
        if key != "session_id":  # Skip session_id as we already handled it
            save_obj_to_zarr(sg, key, val, logger)


def init_or_open_result_store(s3_client: Any, s3_bucket: str, result_key: str, query_name: str, 
                             label_map: Optional[Dict[str, int]] = None,
                             logger: Optional[logging.Logger] = None) -> Tuple[zarr.Group, zarr.Group, bool]:
    """
    Initialize a new query result store or open an existing one.
    
    This function handles all aspects of store initialization:
    1. Checks if the store already exists in S3
    2. Creates a new store with proper attributes if it doesn't exist
    3. Opens and returns an existing store if it does exist
    
    Args:
        s3_client: Boto3 S3 client
        s3_bucket: S3 bucket name
        result_key: S3 key for the zarr store
        query_name: Name of the query
        label_map: Optional mapping for labels (None if query doesn't use labels)
        logger: Optional logger instance
        
    Returns:
        Tuple of (root_group, sessions_group, is_first_session)
    """
    log = logger.info if logger else print
    debug = logger.debug if logger else print
    
    uri = f"s3://{s3_bucket}/{result_key}"
    debug(f"Checking for zarr store at: {uri}")
    
    # Check if this is the first session by looking for zarr.json
    is_first_session = False
    try:
        s3_client.head_object(Bucket=s3_bucket, Key=f"{result_key}/zarr.json")
        debug(f"Found existing zarr store at: {uri}")
        is_first_session = False
    except s3_client.exceptions.ClientError:
        debug(f"No existing zarr store found at: {uri}, will initialize new store")
        is_first_session = True
    
    if is_first_session:
        # Initialize the zarr store
        debug(f"Creating new zarr store at: {uri}")
        try:
            root = open_zarr_safely(uri, mode="w", logger=logger)
            
            # Set attributes
            root.attrs.update(
                query_name=query_name,
                created_at=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                version="0.2",
                session_count=0,
                session_ids=[],
            )
            
            # Only include label_map if provided
            if label_map is not None:
                debug(f"Adding label_map to store attributes: {label_map}")
                root.attrs["label_map"] = label_map
            
            # Create sessions group
            debug("Creating sessions group")
            sessions_group = root.require_group("sessions")
            
        except Exception as e:
            log(f"Error creating zarr store: {type(e).__name__}: {e}")
            raise
            
    else:
        # Open existing store
        debug(f"Opening existing zarr store at: {uri}")
        try:
            root = open_zarr_safely(uri, mode="a", logger=logger)
            # Get existing sessions group
            sessions_group = root["sessions"]
            
        except Exception as e:
            log(f"Error opening zarr store: {type(e).__name__}: {e}")
            raise
    
    return root, sessions_group, is_first_session


def write_session_result(root: zarr.Group, session_result: Dict[str, Any], 
                        session_id: str, logger: Optional[logging.Logger] = None):
    """
    Update the root group's attributes after adding a session.
    
    Args:
        root: Root zarr group
        session_result: Session data that was added
        session_id: Session ID
        logger: Optional logger instance
    """
    log = logger.info if logger else print
    debug = logger.debug if logger else print
    
    import time
    
    # Update root attrs
    ids = set(root.attrs.get("session_ids", []))
    ids.add(session_id)
    root.attrs["session_ids"] = sorted(ids)
    root.attrs["session_count"] = len(ids)
    root.attrs["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    
    # Consolidate metadata
    zarr.consolidate_metadata(root.store)


def extract_elements_data(zgroup: zarr.Group, session_id: str, 
                         logger: Optional[logging.Logger] = None) -> Optional[Dict[str, np.ndarray]]:
    """
    Extract elements data from zarr group.
    
    Args:
        zgroup: Zarr group containing elements data
        session_id: Session ID being processed
        logger: Optional logger instance
        
    Returns:
        Dictionary of element data arrays or None if no elements found
    """
    log = logger.info if logger else print
    debug = logger.debug if logger else print
    warning = logger.warning if logger else print
    
    debug(f"Extracting elements for session: {session_id}")
    debug(f"zgroup type: {type(zgroup)}")
    debug(f"zgroup array keys: {list(zgroup.array_keys())}")
    debug(f"zgroup group keys: {list(zgroup.group_keys())}")
    debug(f"Direct 'elements' check: {'elements' in zgroup}")
    debug(f"Elements in group keys?: {'elements' in list(zgroup.group_keys())}")
    
    if "elements" not in zgroup:
        warning(f"No elements subgroup in {session_id}")
        return None

    eg = zgroup["elements"]
    result: Dict[str, np.ndarray] = {"session_id": session_id}

    # Get element IDs first
    if "element_id" in eg:
        ids = bytesify(eg["element_id"][:])
    else:
        # Create default numeric IDs if not available
        ids = np.arange(eg[list(eg.array_keys())[0]].shape[0])
    result["element_ids"] = ids

    # Extract all element variables
    for name in eg.array_keys():
        # Get data and convert object/unicode to bytes
        value = eg[name][:]
        result[name] = bytesify(value)

    return result


def extract_elements_neural_windows(
    event_group: zarr.Group, 
    window_group: zarr.Group,
    session_id: str, 
    filter_kwargs: Dict[str, Any],
    label_map: Optional[Dict[str, int]] = None,
    logger: Optional[logging.Logger] = None
) -> Optional[Dict[str, np.ndarray]]:
    """
    Extract neural windows during elements that match specified filters.
    
    Args:
        event_group: Zarr group containing elements data
        window_group: Zarr group containing window data
        session_id: Session ID
        filter_kwargs: Dictionary of keyword arguments for filter_elements
        label_map: Optional mapping from element_id patterns to numeric labels
        logger: Optional logger instance
        
    Returns:
        Dictionary with windows and metadata
    """
    log = logger.info if logger else print
    debug = logger.debug if logger else print
    
    # Get filtered elements
    all_elems = extract_elements_data(event_group, session_id, logger)
    if all_elems is None:
        return None
        
    filtered_elems = filter_elements(all_elems, **filter_kwargs)
    if filtered_elems is None:
        return None
    
    log(f"Found {len(filtered_elems['element_ids'])} matching elements for session {session_id}")
    
    # Add debug print for element durations
    debug(
        f"Found matching elements with durations: " + 
        ", ".join([f"{end-start}ms" for start, end in zip(filtered_elems['start_time'], filtered_elems['end_time'])])[:100] + 
        "..."
    )
    
    # Use default label map if none provided
    if label_map is None:
        label_map = DEFAULT_LABEL_MAP
    
    # Get timestamps array
    t = window_group["time"][:]
    debug(f"Time array shape: {window_group['time'].shape}")
    
    # Add debug print for time ranges
    debug(
        f"Session {session_id}: Element time ranges {filtered_elems['start_time'].min()}-{filtered_elems['end_time'].max()} vs "
        f"Window timestamps {t.min()}-{t.max()} (length={len(t)})"
    )
    
    # Find all hits within the element time ranges
    hits = []
    for eid, (start, end) in zip(filtered_elems["element_ids"],
                               zip(filtered_elems["start_time"], filtered_elems["end_time"])):
        window_indices = window_indices_for_range(t, start, end)
        if window_indices.size > 0:
            hits.extend(window_indices)
    
    if not hits:
        debug(f"No window hits found for the filtered elements")
        return None
        
    # Sort for chronological order
    hits = np.sort(np.array(hits))
    
    # Add debug print
    debug(f"Generated {len(hits)} window indices, min={min(hits)}, max={max(hits)}")
    if len(hits) > 1000:
        debug(f"Large number of hits - first 5: {hits[:5]}, last 5: {hits[-5:]}")
    
    log(f"Found {len(hits)} windows across {len(filtered_elems['element_ids'])} elements for session {session_id}")
    
    # Use extract_full_windows to get all variables for the hits
    result = extract_full_windows(window_group, hits)
    
    # Add session_id and generate labels and element_ids
    result.update({
        "session_id": session_id,
        "labels": bytesify(build_labels(filtered_elems, hits, t, label_map)),
        "element_ids": bytesify(build_eids(filtered_elems, hits, t))
    })
    
    return result