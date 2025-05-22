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
    • Generic query registry system for basic element and token queries

Note: For neural-related queries:
    • Use t4B_query_eye_neural_v0.py for eye neural window extraction
    • Use t4C_query_neuro_lang_v0.py for language-neural window extraction
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

# Define default label map for eye task elements
DEFAULT_EYE_LABEL_MAP = {"close": 0, "open": 1, "intro": 2, "unknown": 3}

# Module logger
logger = logging.getLogger(__name__)


class QueryTransform(BaseTransform):
    """
    Query / aggregation stage operating on event Zarr stores.
    
    Note: For neural-related queries:
    - Use t4B_query_eye_neural_v0.py for eye neural window extraction
    - Use t4C_query_neuro_lang_v0.py for language-neural window extraction
    """

    SOURCE_PREFIX = "processed/event/"
    DEST_PREFIX = "processed/queries/"

    # Registry of available queries mapped to their implementation functions
    QUERY_REGISTRY = {
        "all_elements": "_query_all_elements",
        "eye_elements": "_query_eye_elements",
        "lang_tokens": "_query_lang_tokens",
        # Additional queries will be added here
    }
    
    # List of queries that need label maps
    QUERIES_WITH_LABELS = {}

    def __init__(self, query_name: Optional[str] = None, 
                 label_map: Optional[Dict[str, int]] = None,
                 **kwargs):
        """
        Initialize the query transform.
        
        Args:
            query_name: Name of the query to run (must be in QUERY_REGISTRY)
            label_map: Optional mapping from element_id patterns to labels
            **kwargs: Additional arguments for BaseTransform
        """
        self.query_name = query_name
        
        # Only set label_map if this query type uses labels
        self.label_map = None
        if query_name in self.QUERIES_WITH_LABELS:
            self.label_map = label_map or DEFAULT_EYE_LABEL_MAP
        
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
        transform_id = f"t4A_query_v0_{args.query_name}" if args.query_name else "t4A_query_v0"
                
        return cls(
            query_name=args.query_name,
            label_map=label_map,
            transform_id=transform_id,  # Use query-specific transform_id
            source_prefix=getattr(args, "source_prefix", cls.SOURCE_PREFIX),
            destination_prefix=getattr(args, "dest_prefix", cls.DEST_PREFIX),
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
        # Select source prefix based on query type
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




# -------------------------------------------------------------------------- #
#  CLI entry-point
# -------------------------------------------------------------------------- #

if __name__ == "__main__":
    QueryTransform.run_from_command_line()