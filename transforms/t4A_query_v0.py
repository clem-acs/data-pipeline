"""
T4A Query Transform – version 0.2

Cross-session queries on event Zarr v3 stores.
Outputs a single consolidated v3 store per query under:
    processed/queries/<query_name>.zarr

Key features:
    • Zarr-v3-safe writes (create_dataset with explicit shape/dtype)
    • Robust first-session detection (checks '<store>/zarr.json')
    • Object/Unicode dtypes converted to fixed-width bytes
    • Helper utilities (bytesify, filter_elements, extract_full_windows, etc.)
    • Metadata consolidation with root.store
    • Generic query registry system
    • Element-based neural window extraction with configurable labels
"""

import json
import logging
import time
import os
from typing import Dict, Any, List, Optional, Tuple, Union

import numpy as np
import zarr

# Import base transform
from base_transform import BaseTransform, Session

# Import helpers
from transforms.query_helpers import (
    # Constants
    DEFAULT_LABEL_MAP,
    
    # Helper functions
    bytesify, 
    sanitize_session_id, 
    filter_elements, 
    extract_full_windows,
    build_labels,
    build_eids,
    window_indices_for_range,
    
    # Extracted functions
    open_zarr_safely,
    save_obj_to_zarr,
    save_session_to_subgroup,
    init_or_open_result_store,
    write_session_result,
    extract_elements_data,
    extract_elements_neural_windows
)

# Module logger
logger = logging.getLogger(__name__)


class QueryTransform(BaseTransform):
    """Query / aggregation stage operating on event Zarr stores."""

    SOURCE_PREFIX = "processed/event/"
    DEST_PREFIX = "processed/queries/"

    # Registry of available queries mapped to their implementation functions
    QUERY_REGISTRY = {
        "all_elements": "_query_all_elements",
        "eye_elements": "_query_eye_elements",
        "eye_neural": "_query_eye_neural_windows",
        "lang_tokens": "_query_lang_tokens",
        "neuro_lang": "_query_neuro_lang_windows",  # Neural-language windows query
        # Additional queries will be added here
    }
    
    # List of queries that need label maps
    QUERIES_WITH_LABELS = {"eye_neural"}

    def __init__(self, query_name: Optional[str] = None, 
                 label_map: Optional[Dict[str, int]] = None,
                 lang_group: str = "W",
                 tokenizer: str = "gpt2",
                 pre_window_seconds: float = 10.0,
                 **kwargs):
        """
        Initialize the query transform.
        
        Args:
            query_name: Name of the query to run (must be in QUERY_REGISTRY)
            label_map: Optional mapping from element_id patterns to labels
            lang_group: Language group to process (L, W, R, S, W_corrected)
            tokenizer: Tokenizer used in language transform (e.g., gpt2)
            pre_window_seconds: Seconds to include before token start
            **kwargs: Additional arguments for BaseTransform
        """
        self.query_name = query_name
        
        # Only set label_map if this query type uses labels
        self.label_map = None
        if query_name in self.QUERIES_WITH_LABELS:
            self.label_map = label_map or DEFAULT_LABEL_MAP
            
        self.lang_group = lang_group
        self.tokenizer = tokenizer
        self.pre_window_seconds = pre_window_seconds
        
        # Set default transform info if not provided
        transform_id = kwargs.pop('transform_id', 't4A_query_v0')
        script_id = kwargs.pop('script_id', '4A')
        script_name = kwargs.pop('script_name', 'query')
        script_version = kwargs.pop('script_version', 'v0')
        
        # Initialize base class
        super().__init__(
            transform_id=transform_id,
            script_id=script_id,
            script_name=script_name,
            script_version=script_version,
            **kwargs
        )
        
        if self.query_name:
            self.logger.info(f"Initialized QueryTransform for query '{self.query_name}'")
            # Only log label map for queries that actually use it
            if self.label_map:
                self.logger.info(f"Using label map: {self.label_map}")

    @classmethod
    def add_subclass_arguments(cls, parser):
        """
        Add query-specific arguments to the command line parser.
        
        Automatically creates flags for each query in QUERY_REGISTRY.
        """
        group = parser.add_mutually_exclusive_group(required=True)
        for q in cls.QUERY_REGISTRY:
            group.add_argument(
                f"--{q.replace('_', '-')}", 
                action="store_const", 
                const=q, 
                dest="query_name",
                help=f"Run the {q} query"
            )
            
        # Add arguments for label mapping
        parser.add_argument(
            "--label-map", 
            type=str,
            help="Label mapping in JSON format, e.g., '{\"closed\":0,\"open\":1,\"intro\":2}'"
        )
        
        # Add neural language query options
        neuro_lang_group = parser.add_argument_group('neuro-lang options')
        neuro_lang_group.add_argument(
            "--lang-group", 
            type=str, 
            choices=["L", "R", "W", "S", "W_corrected"],
            default="W", 
            help="Language group to process"
        )
        neuro_lang_group.add_argument(
            "--tokenizer", 
            type=str, 
            default="gpt2", 
            help="Tokenizer used in language transform"
        )
        neuro_lang_group.add_argument(
            "--pre-window-seconds", 
            type=float, 
            default=10.0, 
            help="Seconds to include before token start"
        )

    @classmethod
    def from_args(cls, args):
        """
        Create instance from command line arguments.
        
        Args:
            args: Parsed command line arguments
            
        Returns:
            Initialized QueryTransform instance
        """
        # Parse label map if provided
        label_map = None
        if hasattr(args, "label_map") and args.label_map:
            try:
                label_map = json.loads(args.label_map)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON for label-map: {e}")
        
        # Create a query-specific transform_id to track each query type separately
        if args.query_name == "neuro_lang":
            # Include all parameters in transform_id for neuro_lang query
            transform_id = f"t4A_query_v0_{args.query_name}_{args.lang_group}_{args.tokenizer}_{args.pre_window_seconds}s"
        else:
            transform_id = f"t4A_query_v0_{args.query_name}" if args.query_name else "t4A_query_v0"
                
        return cls(
            query_name=args.query_name,
            label_map=label_map,
            transform_id=transform_id,  # Use query-specific transform_id
            source_prefix=getattr(args, "source_prefix", cls.SOURCE_PREFIX),
            destination_prefix=getattr(args, "dest_prefix", cls.DEST_PREFIX),
            lang_group=getattr(args, "lang_group", "W"),
            tokenizer=getattr(args, "tokenizer", "gpt2"),
            pre_window_seconds=getattr(args, "pre_window_seconds", 10.0),
            s3_bucket=args.s3_bucket,
            verbose=args.verbose,
            log_file=args.log_file,
            dry_run=args.dry_run,
            keep_local=getattr(args, "keep_local", False),
        )

    # ------------------------------------------------------------------ #
    #  Core I/O helpers - All implementation moved to query_helpers.py
    # ------------------------------------------------------------------ #

    # Note: _open_zarr_safely, _extract_elements_data, _save_to_zarr, and _save_session_to_subgroup
    # have been replaced with imported helpers from query_helpers.py

    # Note: _save_session_result removed - replaced with direct calls to helper functions

    # ------------------------------------------------------------------ #
    #  BaseTransform hooks
    # ------------------------------------------------------------------ #

    def find_sessions(self) -> List[str]:
        """
        Find sessions with zarr stores matching the query type.
        
        Returns:
            List of session IDs with relevant data
        """
        # For neuro_lang query, need both event and specific tokenizer language zarr stores
        if self.query_name == "neuro_lang":
            # Find sessions with event zarr stores
            event_sessions = set()
            resp = self.s3.list_objects_v2(Bucket=self.s3_bucket, Prefix=self.source_prefix)
            for obj in resp.get("Contents", []):
                key = obj["Key"]
                if key.endswith("_events.zarr/zarr.json"):
                    session_id = key.split("/")[-2].replace("_events.zarr", "")
                    event_sessions.add(session_id)
            
            # Find sessions with specific tokenizer lang zarr stores
            lang_prefix = "processed/lang/"
            tokenizer_sessions = set()
            resp = self.s3.list_objects_v2(Bucket=self.s3_bucket, Prefix=lang_prefix)
            for obj in resp.get("Contents", []):
                key = obj["Key"]
                if key.endswith("_lang.zarr/zarr.json"):
                    # Check if this has our specific tokenizer
                    zarr_name = key.split("/")[-2]  # e.g., "session123_gpt2_lang.zarr"
                    if f"_{self.tokenizer}_lang.zarr" in zarr_name:
                        # Extract session ID from the tokenizer zarr name (without _events.zarr suffix)
                        session_id = zarr_name.replace(f"_{self.tokenizer}_lang.zarr", "")
                        # Add both clean ID and ID with _events.zarr suffix to handle both path formats
                        tokenizer_sessions.add(session_id)
                        # Also add with _events.zarr suffix to match event session IDs
                        tokenizer_sessions.add(f"{session_id}_events.zarr")
            
            # Find sessions with window zarr stores
            window_prefix = "processed/windows/"
            window_sessions = set()
            resp = self.s3.list_objects_v2(Bucket=self.s3_bucket, Prefix=window_prefix)
            for obj in resp.get("Contents", []):
                key = obj["Key"]
                if key.endswith("_windowed.zarr/zarr.json"):
                    session_id = key.split("/")[-2].replace("_windowed.zarr", "")
                    window_sessions.add(session_id)
            
            # Return sessions with event, specific tokenizer lang, and window data
            common_sessions = event_sessions.intersection(tokenizer_sessions).intersection(window_sessions)
            self.logger.info(f"Found {len(common_sessions)} sessions with event zarr, {self.tokenizer} language zarr, and window zarr stores")
            return sorted(common_sessions)
        
        # Select source prefix based on query type for other queries
        prefix = self.source_prefix
        
        # Use lang prefix if it's a lang query
        if self.query_name == "lang_tokens":
            prefix = "processed/lang/"
            self.logger.info(f"Scanning {prefix} for *_lang.zarr")
            pattern = "_lang.zarr/zarr.json"
            replacer = "_lang.zarr"
        else:
            # Default to event zarr files
            self.logger.info(f"Scanning {prefix} for *_events.zarr")
            pattern = "_events.zarr/zarr.json"
            replacer = "_events.zarr"
            
        # List objects
        resp = self.s3.list_objects_v2(Bucket=self.s3_bucket, Prefix=prefix)
        
        # Debug log
        self.logger.debug(f"All keys found: {[obj['Key'] for obj in resp.get('Contents', [])]}")
        
        hits = set()
        for obj in resp.get("Contents", []):
            key = obj["Key"]
            self.logger.debug(f"Checking key: {key}")
            
            if key.endswith(pattern):
                # Extract session ID from the key
                session_id = key.split("/")[-2].replace(replacer, "")
                
                # For lang stores, handle tokenizer in the name
                if "_" in session_id and self.query_name == "lang_tokens":
                    # Format is {session_id}_{tokenizer}_lang.zarr
                    parts = session_id.split("_")
                    if len(parts) > 1:
                        # Use only the session ID part, not the tokenizer
                        session_id = "_".join(parts[:-1])
                
                self.logger.info(f"Found session ID: {session_id}")
                hits.add(session_id)
                
        return sorted(hits)

    def process_session(self, session: Session) -> Dict[str, Any]:
        """
        Process a single session using the selected query.
        
        Args:
            session: Session object
            
        Returns:
            Dict with processing result
        """
        sid = session.session_id
        self.logger.info(f"Processing session with ID: {sid}")
        
        # Determine zarr store key based on query type
        if self.query_name == "lang_tokens":
            # Find all lang zarr stores for this session
            # (may have multiple tokenizers)
            prefix = f"processed/lang/{sid}_"
            resp = self.s3.list_objects_v2(Bucket=self.s3_bucket, Prefix=prefix)
            
            lang_zarrs = []
            for obj in resp.get("Contents", []):
                key = obj["Key"]
                if key.endswith("_lang.zarr/zarr.json"):
                    # Extract store path
                    zarr_path = key.replace("/zarr.json", "")
                    lang_zarrs.append(zarr_path)
            
            if not lang_zarrs:
                self.logger.error(f"No language zarr stores found for {sid}")
                return {"status": "skipped", "metadata": {"reason": "no_lang_zarr"}}
                
            # Use the first lang zarr store found (could process all of them in the future)
            zarr_key = lang_zarrs[0]
            self.logger.info(f"Using lang zarr store: {zarr_key}")
        else:
            # Default to events zarr
            zarr_key = (
                f"{self.source_prefix}{sid}"
                if sid.endswith("_events.zarr")
                else f"{self.source_prefix}{sid}_events.zarr"
            )
            
        self.logger.info(f"Looking for zarr at key: {zarr_key}/zarr.json")
        
        try:
            # Check if zarr exists
            try:
                self.s3.head_object(Bucket=self.s3_bucket, Key=f"{zarr_key}/zarr.json")
            except self.s3.exceptions.ClientError:
                if self.query_name == "lang_tokens":
                    self.logger.error(f"No language zarr for {sid}")
                    return {"status": "skipped", "metadata": {"reason": "no_lang_zarr"}}
                else:
                    self.logger.error(f"No events zarr for {sid}")
                    return {"status": "skipped", "metadata": {"reason": "no_events_zarr"}}
            
            # Open zarr store
            zgroup = open_zarr_safely(f"s3://{self.s3_bucket}/{zarr_key}", logger=self.logger)
            
            # Execute query based on registry
            method_name = self.QUERY_REGISTRY.get(self.query_name)
            if not method_name:
                return {
                    "status": "failed", 
                    "error_details": f"unknown query '{self.query_name}'"
                }
                
            # Call the query method
            result = getattr(self, method_name)(zgroup, sid)
            if result is None:
                return {
                    "status": "skipped", 
                    "metadata": {"reason": "no_matches"}
                }
            
            # Calculate statistics
            if self.query_name == "lang_tokens":
                # For lang queries, count tokens
                token_count = 0
                for group_name in ['L', 'W', 'R', 'S', 'W_corrected']:
                    if group_name in result and 'tokens' in result[group_name]:
                        token_count += len(result[group_name]['tokens'])
                count = token_count
            else:
                # For other queries, use element_ids
                count = len(result.get("element_ids", []))
                
            # Size calculation removed (unused)
            
            # Save results if not dry run
            if not self.dry_run:
                out_key = f"{self.destination_prefix}{self.query_name}.zarr"
                # Use include_processed from BaseTransform to determine overwrite behavior
                overwrite = getattr(self, 'include_processed', False)
                # Use helper functions to save the session result
                uri = f"s3://{self.s3_bucket}/{out_key}"
                
                # Initialize or open the result store using the helper function
                root, sessions_group, is_first_session = init_or_open_result_store(
                    s3_client=self.s3,
                    s3_bucket=self.s3_bucket,
                    result_key=out_key,
                    query_name=self.query_name,
                    label_map=self.label_map if self.query_name in self.QUERIES_WITH_LABELS else None,
                    logger=self.logger
                )
                
                # Save session and update store
                save_session_to_subgroup(sessions_group, sid, result, logger=self.logger, overwrite=overwrite)
                write_session_result(root, result, sid, logger=self.logger)
            
            # Return success
            return {
                "status": "success",
                "metadata": {
                    "session_id": sid,
                    "matches_found": count,
                    "query_type": self.query_name,
                },
                "zarr_stores": [f"{self.destination_prefix}{self.query_name}.zarr"],
            }
            
        except Exception as e:
            self.logger.error(f"Error processing query for {sid}: {e}", exc_info=True)
            return {
                "status": "failed",
                "error_details": str(e),
                "metadata": {"session_id": sid},
            }

    # ------------------------------------------------------------------ #
    #  Query Methods
    # ------------------------------------------------------------------ #

    def _query_all_elements(self, zgroup: zarr.Group, sid: str) -> Optional[Dict[str, np.ndarray]]:
        """
        Extract all elements from a session.
        
        Args:
            zgroup: Zarr group containing elements data
            sid: Session ID
            
        Returns:
            Dictionary with all element data
        """
        return extract_elements_data(zgroup, sid, logger=self.logger)

    def _query_eye_elements(self, zgroup: zarr.Group, sid: str) -> Optional[Dict[str, np.ndarray]]:
        """
        Extract elements with task_type 'eye'.
        
        Args:
            zgroup: Zarr group containing elements data
            sid: Session ID
            
        Returns:
            Dictionary with filtered element data
        """
        elems = extract_elements_data(zgroup, sid, logger=self.logger)
        if not elems:
            return None
            
        # Filter for task_type='eye'
        filtered_elems = filter_elements(elems, task_type="eye")
        
        # Add debug print
        if filtered_elems:
            self.logger.debug(f"Found {len(filtered_elems['element_ids'])} eye elements for session {sid}")
            # Print sample element
            if len(filtered_elems['element_ids']) > 0:
                self.logger.debug(f"Sample eye element: ID={filtered_elems['element_ids'][0]}, time_range={filtered_elems['start_time'][0]}-{filtered_elems['end_time'][0]}")
        else:
            self.logger.debug(f"No eye elements found for session {sid}")
            
        return filtered_elems

    def _query_eye_neural_windows(self, zgroup: zarr.Group, sid: str) -> Optional[Dict[str, np.ndarray]]:
        """
        Extract neural data windows during 'eye' task elements.
        
        Args:
            zgroup: Zarr group containing elements data
            sid: Session ID
            
        Returns:
            Dictionary with windows, labels, and element_ids
        """
        # Clean session ID for window store path (remove _events.zarr suffix if present)
        clean_sid = sid
        if clean_sid.endswith("_events.zarr"):
            clean_sid = clean_sid.replace("_events.zarr", "")
        
        # Open the window store for this session
        window_key = f"processed/windows/{clean_sid}_windowed.zarr"
        
        try:
            self.s3.head_object(Bucket=self.s3_bucket, Key=f"{window_key}/zarr.json")
        except self.s3.exceptions.ClientError:
            self.logger.warning(f"No window store for {sid}")
            return None
        
        # Open window zarr store
        wgroup = open_zarr_safely(f"s3://{self.s3_bucket}/{window_key}", logger=self.logger)
        
        # Use the extracted helper function with task_type="eye" filter
        return extract_elements_neural_windows(
            event_group=zgroup,
            window_group=wgroup,
            session_id=sid,
            filter_kwargs={"task_type": "eye"},
            label_map=self.label_map if self.query_name in self.QUERIES_WITH_LABELS else None,
            logger=self.logger
        )
        
    def _query_lang_tokens(self, zgroup: zarr.Group, sid: str) -> Optional[Dict[str, Any]]:
        """
        Extract language tokens from a session.
        
        Args:
            zgroup: Zarr group containing language data
            sid: Session ID
            
        Returns:
            Dictionary with language token data
        """
        from transforms.query_helpers import extract_lang_tokens
        
        # Get tokenizer from the root attributes
        tokenizer = zgroup.attrs.get('tokenizer', 'unknown')
        
        # Extract tokens from each language group
        result = {
            'session_id': sid,
            'tokenizer': tokenizer
        }
        
        # Extract language groups
        for group_name in ['L', 'W', 'R', 'S', 'W_corrected']:
            group_data = extract_lang_tokens(zgroup, group_name)
            if group_data:
                result[group_name] = group_data
        
        # Only return if we found any language data
        if len(result) > 2:  # More than just session_id and tokenizer
            return result
        return None

    def _query_neuro_lang_windows(self, zgroup: zarr.Group, sid: str) -> Optional[Dict[str, Any]]:
        """
        Extract neural windows for language tokens.
        
        For each token of the specified language group within an element:
        1. Get windows from N seconds before token start to token end
        2. Associate these windows with the token and its element
        
        Args:
            zgroup: Event zarr group with elements data
            sid: Session ID
            
        Returns:
            Dictionary with tokens and their associated neural windows
        """
        # Get elements data from event zarr
        elements = extract_elements_data(zgroup, sid, logger=self.logger)
        if elements is None:
            self.logger.warning(f"No elements found for {sid}")
            return None
            
        # Find and open the language zarr store with specific tokenizer
        # Clean session ID by removing _events.zarr suffix if present
        clean_sid = sid.replace("_events.zarr", "")
        lang_zarr_key = f"processed/lang/{clean_sid}_{self.tokenizer}_lang.zarr"
        try:
            self.s3.head_object(Bucket=self.s3_bucket, Key=f"{lang_zarr_key}/zarr.json")
        except self.s3.exceptions.ClientError:
            self.logger.warning(f"No language zarr for {clean_sid} with tokenizer {self.tokenizer}")
            return None
            
        lang_zgroup = open_zarr_safely(f"s3://{self.s3_bucket}/{lang_zarr_key}", logger=self.logger)
        
        # Extract tokens for the specified language group
        from transforms.query_helpers import extract_lang_tokens, find_element_for_timestamp
        lang_data = extract_lang_tokens(lang_zgroup, self.lang_group)
        if lang_data is None or 'tokens' not in lang_data or not lang_data['tokens']:
            self.logger.warning(f"No {self.lang_group} tokens found for {sid}")
            return None
            
        # Find and open the window zarr store
        window_key = f"processed/windows/{sid.replace('_events.zarr', '')}_windowed.zarr"
        try:
            self.s3.head_object(Bucket=self.s3_bucket, Key=f"{window_key}/zarr.json")
        except self.s3.exceptions.ClientError:
            self.logger.warning(f"No window store for {sid}")
            return None
            
        window_zgroup = open_zarr_safely(f"s3://{self.s3_bucket}/{window_key}", logger=self.logger)
        window_times = window_zgroup["time"][:]
        
        # Create sequence of tokens in chronological order
        tokens = lang_data["tokens"]
        token_sequence = list(range(len(tokens)))
        
        # Sort tokens by start timestamp if available
        if tokens and "start_timestamp" in tokens[0]:
            token_sequence = sorted(token_sequence, 
                                   key=lambda i: tokens[i]["start_timestamp"])
        
        # Initialize arrays for vectorized token data
        all_token_text = []
        all_token_ids = []
        all_start_times = []
        all_end_times = []
        all_special_flags = []
        all_element_ids = []  # Which element each token belongs to

        # Initialize window storage
        all_windows = {}  # Map of window_idx -> window data (for deduplication)
        
        # Initialize token-window mapping
        token_indices = []  # Token indices in our arrays
        window_indices = []  # Window indices

        # Process each token and find its element and windows
        tokens_with_windows_count = 0
        for idx, token_idx in enumerate(token_sequence):
            token = tokens[token_idx]
            
            # Get token timestamps
            start_ts = token.get("start_timestamp", 0)
            end_ts = token.get("end_timestamp", start_ts)
            
            # Find which element this token belongs to
            element_id = find_element_for_timestamp(elements, start_ts)
            if element_id is None:
                continue  # Skip tokens not in any element
                
            # Calculate window start time (N seconds before token)
            window_start = max(0, start_ts - (self.pre_window_seconds * 1000))  # Convert to ms
            
            # Get window indices within the time range
            win_idx_array = window_indices_for_range(window_times, window_start, end_ts)
            
            if not win_idx_array.size:
                continue  # Skip tokens with no windows
                
            # Add token data to arrays
            all_token_text.append(token.get("token", ""))
            all_token_ids.append(token.get("token_id", 0))
            all_start_times.append(start_ts)
            all_end_times.append(end_ts)
            all_special_flags.append(token.get("special_token", False))
            all_element_ids.append(element_id)
            
            # Map this token to its windows
            curr_token_idx = len(all_token_text) - 1  # Index in our arrays
            for window_idx in win_idx_array:
                token_indices.append(curr_token_idx)
                window_indices.append(int(window_idx))
                
                # Store unique window data if not already stored
                if window_idx not in all_windows:
                    # Get window data for this single index
                    from transforms.query_helpers import extract_full_windows
                    # Convert to NumPy array to avoid 'list' object has no attribute 'min' error
                    window_data = extract_full_windows(window_zgroup, np.array([window_idx]))
                    
                    # Store at the integer index for consistent retrieval
                    all_windows[int(window_idx)] = {
                        k: v[0] if isinstance(v, np.ndarray) and v.shape[0] == 1 else v
                        for k, v in window_data.items()
                    }
            
            tokens_with_windows_count += 1
        
        if tokens_with_windows_count == 0:
            self.logger.warning(f"No {self.lang_group} tokens with windows found for {sid}")
            return None
        
        # Convert window dict to vectorized arrays
        window_keys = sorted(all_windows.keys())
        window_arrays = {}
        
        # Get all keys in the window data
        first_window = next(iter(all_windows.values()))
        for key in first_window.keys():
            # Collect all values for this key into an array
            values = [all_windows[idx][key] for idx in window_keys]
            
            # Convert to numpy array
            if all(isinstance(v, np.ndarray) for v in values):
                # Stack arrays along a new first dimension
                stacked = np.stack(values)
                window_arrays[key] = stacked
                # Note: We're not setting chunks directly here because it will be handled
                # by the improved chunking logic in _save_to_zarr
            else:
                # Simple array conversion
                window_arrays[key] = np.array(values)
        
        # Map window indices to positions in our arrays
        window_idx_map = {idx: pos for pos, idx in enumerate(window_keys)}
        remapped_window_indices = [window_idx_map[idx] for idx in window_indices]
            
        # Format final result
        result = {
            "session_id": sid,
            "tokenizer": self.tokenizer,
            "lang_group": self.lang_group,
            "pre_window_seconds": self.pre_window_seconds,
            "tokens": {
                "text": np.array(all_token_text),
                "ids": np.array(all_token_ids),
                "start_times": np.array(all_start_times),
                "end_times": np.array(all_end_times),
                "special": np.array(all_special_flags),
                "element_ids": np.array(all_element_ids)
            },
            "windows": window_arrays,
            "token_window_map": {
                "token_idx": np.array(token_indices),
                "window_idx": np.array(remapped_window_indices)
            }
        }
        
        print(f"\n=== QUERY RESULT FOR {sid} ===")
        print(f"Result keys: {list(result.keys())}")
        print(f"tokens count: {len(result['tokens']['text'])}")
        print(f"window count: {len(window_keys)}")
        print(f"token-window mapping count: {len(token_indices)}")
        
        # Print window data info
        if window_arrays:
            print(f"Window array keys: {list(window_arrays.keys())}")
            for k, v in window_arrays.items():
                if isinstance(v, np.ndarray):
                    print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
        
        return result



# -------------------------------------------------------------------------- #
#  CLI entry-point
# -------------------------------------------------------------------------- #

if __name__ == "__main__":
    QueryTransform.run_from_command_line()