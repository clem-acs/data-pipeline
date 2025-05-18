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
    bytesify, 
    sanitize_session_id, 
    filter_elements, 
    join_elements_tasks,
    collect_windows,
    extract_full_windows,
    build_labels,
    build_eids,
    get_neural_windows_dataset
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
        # Additional queries will be added here
    }

    def __init__(self, query_name: Optional[str] = None, 
                 label_map: Optional[Dict[str, int]] = None, **kwargs):
        """
        Initialize the query transform.
        
        Args:
            query_name: Name of the query to run (must be in QUERY_REGISTRY)
            label_map: Optional mapping from element_id patterns to labels
            **kwargs: Additional arguments for BaseTransform
        """
        self.query_name = query_name
        self.label_map = label_map or {"closed": 0, "open": 1, "intro": 2, "unknown": 3}
        
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
    #  Core I/O helpers
    # ------------------------------------------------------------------ #

    def _open_zarr_safely(self, uri: str) -> zarr.Group:
        """
        Open an S3-hosted Zarr store read-only.
        
        Args:
            uri: URI for the zarr store (e.g., s3://bucket/path/to/store.zarr)
            
        Returns:
            Open zarr.Group object
        """
        return zarr.open_group(
            store=uri, 
            mode="r", 
            storage_options={"anon": False}
        )

    def _extract_elements_data(self, zgroup: zarr.Group, session_id: str) -> Optional[Dict[str, np.ndarray]]:
        """
        Extract elements data from zarr group.
        
        Args:
            zgroup: Zarr group containing elements data
            session_id: Session ID being processed
            
        Returns:
            Dictionary of element data arrays or None if no elements found
        """
        if "elements" not in zgroup:
            self.logger.warning(f"No elements subgroup in {session_id}")
            return None

        eg = zgroup["elements"]
        result: Dict[str, np.ndarray] = {"session_id": session_id}

        # Get element IDs first
        if "element_id" in zgroup:
            ids = bytesify(zgroup["element_id"][:])
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

    def _save_session_to_subgroup(
        self,
        sessions_group: zarr.Group,
        session_id: str,
        data: Dict[str, Any],
    ):
        """
        Save session data to a session-specific subgroup.
        
        Args:
            sessions_group: Parent zarr group for all sessions
            session_id: Session ID being processed
            data: Dictionary with query results
        """
        # Create/get session-specific subgroup
        sg = sessions_group.require_group(sanitize_session_id(session_id))
        
        # Save each item
        for key, val in data.items():
            if key == "session_id":  # Skip, handled in metadata
                sg.attrs["session_id"] = val
                continue
                
            if isinstance(val, np.ndarray):
                # Ensure bytes-safe dtype
                val = bytesify(val)
                
                # Determine appropriate chunking for the array
                # For multi-dimensional arrays > 2GB, cap chunk size
                chunks = val.shape
                if val.nbytes > 2e9 and val.ndim > 1:
                    # For 1D array, cap at 10,000 elements per chunk
                    if val.ndim == 1:
                        chunks = (min(10_000, val.shape[0]),)
                    # For higher-dim arrays, cap only first dimension
                    else:
                        max_chunk_0 = max(1, int(2e9 / (val.itemsize * np.prod(val.shape[1:]))))
                        chunks = (min(max_chunk_0, val.shape[0]),) + val.shape[1:]
                
                # Create dataset with explicit shape, dtype, chunks
                sg.create_dataset(
                    name=key,
                    data=val,
                    shape=val.shape,
                    dtype=val.dtype,
                    chunks=chunks
                )
            elif isinstance(val, (str, int, float, bool)):
                # Store scalar values as attributes
                sg.attrs[key] = val

    def _save_session_result(
        self,
        session_result: Dict[str, Any],
        result_key: str,
        session_id: str,
    ):
        """
        Save a session's result to the zarr store.
        
        Handles first-session initialization and metadata management.
        
        Args:
            session_result: Dictionary with this session's query results
            result_key: S3 key for the zarr store
            session_id: Session ID being processed
        """
        uri = f"s3://{self.s3_bucket}/{result_key}"
        
        # Is this the first session? Check for zarr.json file
        try:
            self.s3.head_object(Bucket=self.s3_bucket, Key=f"{result_key}/zarr.json")
            first = False
            self.logger.info(f"Appending results for session {session_id} to {uri}")
        except self.s3.exceptions.ClientError:
            first = True
            self.logger.info(f"Initializing new zarr store at {uri}")

        try:
            if first:
                # Initialize the zarr store
                root = zarr.group(store=uri, storage_options={"anon": False})
                root.attrs.update(
                    query_name=self.query_name,
                    created_at=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                    version="0.2",
                    session_count=0,
                    session_ids=[],
                )
                sessions_group = root.require_group("sessions")
            else:
                # Open existing store
                root = zarr.open_group(store=uri, mode="a", storage_options={"anon": False})
                sessions_group = root["sessions"]
            
            # Create backup of critical metadata for transaction-like safety
            backup = {
                'attrs': dict(root.attrs)
            }
            
            # Write session subgroup
            self._save_session_to_subgroup(sessions_group, session_id, session_result)
            
            # Update root attrs
            ids = set(root.attrs.get("session_ids", []))
            ids.add(session_id)
            root.attrs["session_ids"] = sorted(ids)
            root.attrs["session_count"] = len(ids)
            root.attrs["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            
            # Consolidate metadata
            zarr.consolidate_metadata(root.store)
            
        except Exception as e:
            self.logger.error(f"Error saving results for {session_id}: {e}")
            
            # Attempt to rollback changes on failure
            if not first:
                try:
                    # Restore attributes if backup exists
                    if 'backup' in locals():
                        root = zarr.open_group(store=uri, mode="a", storage_options={"anon": False})
                        sessions_group = root["sessions"]
                        
                        # Restore attributes
                        for k, v in backup['attrs'].items():
                            root.attrs[k] = v
                        
                        # Remove session subgroup if it was created
                        session_key = sanitize_session_id(session_id)
                        if session_key in sessions_group:
                            del sessions_group[session_key]
                            
                        self.logger.info("Rollback completed successfully")
                except Exception as rollback_error:
                    self.logger.error(f"Error during rollback: {rollback_error}")
            
            # Re-raise the original exception
            raise e

    # ------------------------------------------------------------------ #
    #  BaseTransform hooks
    # ------------------------------------------------------------------ #

    def find_sessions(self) -> List[str]:
        """
        Find sessions with event zarr stores.
        
        Returns:
            List of session IDs with event data
        """
        self.logger.info(f"Scanning {self.source_prefix} for *_events.zarr")
        self.logger.info(f"SOURCE_PREFIX: {self.SOURCE_PREFIX}, source_prefix: {self.source_prefix}")
        resp = self.s3.list_objects_v2(Bucket=self.s3_bucket, Prefix=self.source_prefix)
        # Print all keys for debugging
        self.logger.info(f"All keys found: {[obj['Key'] for obj in resp.get('Contents', [])]}")
        
        hits = set()
        for obj in resp.get("Contents", []):
            key = obj["Key"]
            self.logger.info(f"Checking key: {key}")
            if key.endswith("_events.zarr/zarr.json"):
                # Extract session ID from the key
                session_id = key.split("/")[-2].replace("_events.zarr", "")
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
        
        zarr_key = (
            f"{self.source_prefix}{sid}"
            if sid.endswith("_events.zarr")
            else f"{self.source_prefix}{sid}_events.zarr"
        )
        self.logger.info(f"Looking for zarr at key: {zarr_key}/zarr.json")
        
        try:
            # Check if event zarr exists
            try:
                self.s3.head_object(Bucket=self.s3_bucket, Key=f"{zarr_key}/zarr.json")
            except self.s3.exceptions.ClientError:
                self.logger.error(f"No events zarr for {sid}")
                return {"status": "skipped", "metadata": {"reason": "no_events_zarr"}}
            
            # Open zarr store
            zgroup = self._open_zarr_safely(f"s3://{self.s3_bucket}/{zarr_key}")
            
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
            count = len(result.get("element_ids", []))
            size_mb = sum(
                v.nbytes for v in result.values() 
                if isinstance(v, np.ndarray)
            ) / 1_048_576
            
            # Save results if not dry run
            if not self.dry_run:
                out_key = f"{self.destination_prefix}{self.query_name}.zarr"
                self._save_session_result(result, out_key, sid)
            
            # Return success
            return {
                "status": "success",
                "metadata": {
                    "session_id": sid,
                    "matches_found": count,
                    "size_mb": size_mb,
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
        return self._extract_elements_data(zgroup, sid)

    def _query_eye_elements(self, zgroup: zarr.Group, sid: str) -> Optional[Dict[str, np.ndarray]]:
        """
        Extract elements with task_type 'eye'.
        
        Args:
            zgroup: Zarr group containing elements data
            sid: Session ID
            
        Returns:
            Dictionary with filtered element data
        """
        elems = self._extract_elements_data(zgroup, sid)
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

    def _query_elements_neural_windows(
        self, 
        zgroup: zarr.Group, 
        sid: str, 
        filter_kwargs: Dict[str, Any],
        label_map: Optional[Dict[str, int]] = None
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        Extract neural windows during elements that match specified filters.
        
        Args:
            zgroup: Zarr group containing elements data
            sid: Session ID
            filter_kwargs: Dictionary of keyword arguments for filter_elements
            label_map: Optional mapping from element_id patterns to numeric labels
            
        Returns:
            Dictionary with windows and metadata
        """
        # Get filtered elements
        all_elems = self._extract_elements_data(zgroup, sid)
        if all_elems is None:
            return None
            
        filtered_elems = filter_elements(all_elems, **filter_kwargs)
        if filtered_elems is None:
            return None
        
        logging.info(f"Found {len(filtered_elems['element_ids'])} matching elements for session {sid}")
        
        # Add debug print for element durations
        self.logger.debug(
            f"Found {len(filtered_elems['element_ids'])} matching eye elements with durations: " + 
            ", ".join([f"{end-start}ms" for start, end in zip(filtered_elems['start_time'], filtered_elems['end_time'])])[:100] + 
            "..."
        )

        # Use label map from instance if not provided
        if label_map is None:
            label_map = self.label_map

        # Clean session ID for window store path (remove _events.zarr suffix if present)
        clean_sid = sid
        if clean_sid.endswith("_events.zarr"):
            clean_sid = clean_sid.replace("_events.zarr", "")
        
        # Open the window store for this session
        window_key = f"processed/windows/{clean_sid}_windowed.zarr"
        logging.info(f"Looking for window store at: {window_key}")
        
        # Add debug print
        self.logger.debug(f"Window store path: {window_key}")
        try:
            self.s3.head_object(Bucket=self.s3_bucket, Key=f"{window_key}/zarr.json")
            self.logger.debug(f"Window zarr store found at {window_key}")
        except self.s3.exceptions.ClientError as e:
            self.logger.debug(f"Window zarr store NOT found: {e}")
            self.logger.warning(f"No window store for {sid}")
            return None
        
        # Open window zarr store
        wgroup = self._open_zarr_safely(f"s3://{self.s3_bucket}/{window_key}")
        
        # Add debug print
        self.logger.debug(f"Window zarr array keys: {list(wgroup.array_keys())}")
        
        # Get timestamps array
        t = wgroup["time"][:]
        self.logger.debug(f"Time array shape: {wgroup['time'].shape}")
        
        # Add debug print for time ranges
        self.logger.debug(
            f"Session {sid}: Element time ranges {filtered_elems['start_time'].min()}-{filtered_elems['end_time'].max()} vs "
            f"Window timestamps {t.min()}-{t.max()} (length={len(t)})"
        )
        
        # Find all hits within the element time ranges
        hits = []
        for eid, (start, end) in zip(filtered_elems["element_ids"],
                                   zip(filtered_elems["start_time"], filtered_elems["end_time"])):
            window_indices = np.where((t >= start) & (t <= end))[0]
            if window_indices.size > 0:
                hits.extend(window_indices)
        
        if not hits:
            self.logger.debug(f"No window hits found for eye elements")
            return None
            
        # Sort for chronological order
        hits = np.sort(np.array(hits))
        
        # Add debug print
        self.logger.debug(f"Generated {len(hits)} window indices, min={min(hits)}, max={max(hits)}")
        if len(hits) > 1000:
            self.logger.debug(f"Large number of hits - first 5: {hits[:5]}, last 5: {hits[-5:]}")
        
        logging.info(f"Found {len(hits)} windows across {len(filtered_elems['element_ids'])} elements for session {sid}")
        
        # Use extract_full_windows to get all variables for the hits
        result = extract_full_windows(wgroup, hits)
        
        # Add session_id and generate labels and element_ids
        result.update({
            "session_id": sid,
            "labels": bytesify(build_labels(filtered_elems, hits, t, label_map)),
            "element_ids": bytesify(build_eids(filtered_elems, hits, t))
        })
        
        return result
        
    def _query_eye_neural_windows(self, zgroup: zarr.Group, sid: str) -> Optional[Dict[str, np.ndarray]]:
        """
        Extract neural data windows during 'eye' task elements.
        
        Args:
            zgroup: Zarr group containing elements data
            sid: Session ID
            
        Returns:
            Dictionary with windows, labels, and element_ids
        """
        # Use the generic method with task_type="eye" filter
        return self._query_elements_neural_windows(
            zgroup=zgroup,
            sid=sid,
            filter_kwargs={"task_type": "eye"},
            label_map=self.label_map
        )

    # ------------------------------------------------------------------ #
    #  PyTorch Dataset Conversion
    # ------------------------------------------------------------------ #

    def get_torch_dataset(self, query_result_path: str):
        """
        Load a query result and convert it to a PyTorch dataset.
        
        Args:
            query_result_path: Path to the query result zarr store
            
        Returns:
            torch.utils.data.TensorDataset containing all data
        """
        import torch
        
        # Open the zarr store
        self.logger.info(f"Loading query result from {query_result_path}")
        root = self._open_zarr_safely(query_result_path)
        
        # Check if this is a properly structured query result
        if "sessions" not in root:
            self.logger.error(f"Invalid query result format at {query_result_path}")
            return None
            
        sessions_group = root["sessions"]
        session_ids = list(sessions_group)
        
        if not session_ids:
            self.logger.error(f"No sessions found in {query_result_path}")
            return None
            
        self.logger.info(f"Found {len(session_ids)} sessions in query result")
        
        # Look at first session to determine query type
        first_session = sessions_group[session_ids[0]]
        
        # Handle neural windows query
        if "windows" in first_session and "labels" in first_session:
            return get_neural_windows_dataset(sessions_group, session_ids)
        # Handle elements query 
        elif "element_ids" in first_session:
            self.logger.error("Element queries are not yet supported for PyTorch conversion")
            return None
        else:
            self.logger.error(f"Unknown query type at {query_result_path}")
            return None


# -------------------------------------------------------------------------- #
#  CLI entry-point
# -------------------------------------------------------------------------- #

if __name__ == "__main__":
    QueryTransform.run_from_command_line()