"""
T4B Eye Neural Transform – version 0.0

Single-query transform that extracts neural window data during eye task
elements and aggregates them into a consolidated Zarr v3 store at:
    processed/queries/eye_neural.zarr
"""

import json
import logging
from typing import Any, Dict, List, Optional, Set, Union

import numpy as np
import zarr

from base_transform import BaseTransform, Session
from transforms.query_helpers import (
    open_zarr_safely,
    init_or_open_result_store,
    save_session_to_subgroup,
    write_session_result,
    extract_elements_neural_windows,
)

# Define default label map for eye task elements
DEFAULT_EYE_LABEL_MAP = {"close": 0, "open": 1, "intro": 2, "unknown": 3}

logger = logging.getLogger(__name__)


class EyeNeuralTransform(BaseTransform):
    """
    Extract neural data windows that occur during 'eye' task elements.
    
    This transform:
    1. Finds sessions with both event and window zarr data
    2. Extracts neural windows that occur during elements with task_type='eye'
    3. Labels windows using the provided label_map (pattern matching on element_ids)
    4. Consolidates the data into a single zarr store at processed/queries/eye_neural.zarr
    """

    SOURCE_PREFIX = "processed/event/"
    DEST_PREFIX = "processed/queries/"

    def __init__(self, label_map: Optional[Dict[str, int]] = None, **kwargs):
        """
        Initialize the Eye Neural transform.
        
        Args:
            label_map: Optional mapping of element pattern to numeric label. 
                       If not provided, uses DEFAULT_EYE_LABEL_MAP.
                       Format: {"pattern1": 0, "pattern2": 1, ...}
                       Patterns are matched against element_ids.
            **kwargs: Additional arguments for BaseTransform
        """
        self.label_map = label_map or DEFAULT_EYE_LABEL_MAP

        transform_id = kwargs.pop("transform_id", "t4B_eye_neural_v0")
        script_id = kwargs.pop("script_id", "4B")
        script_name = kwargs.pop("script_name", "eye_neural")
        script_version = kwargs.pop("script_version", "v0")

        super().__init__(
            transform_id=transform_id,
            script_id=script_id,
            script_name=script_name,
            script_version=script_version,
            **kwargs,
        )

        self.logger.info(f"Initialized EyeNeuralTransform with label_map={self.label_map}")

    # ------------------------------------------------------------------ #
    # CLI helpers
    # ------------------------------------------------------------------ #

    @classmethod
    def add_subclass_arguments(cls, parser) -> None:
        """
        Add transform-specific command-line arguments.
        
        Args:
            parser: ArgumentParser instance
        """
        parser.add_argument(
            "--label-map",
            type=str,
            help="Label mapping in JSON, e.g. '{\"closed\":0,\"open\":1,\"intro\":2}'. "
                 "These patterns are matched against element_ids to determine the "
                 "label for each window.",
        )

    @classmethod
    def from_args(cls, args) -> 'EyeNeuralTransform':
        """
        Create transform instance from command-line arguments.
        
        Args:
            args: Parsed command-line arguments
            
        Returns:
            Configured EyeNeuralTransform instance
            
        Raises:
            ValueError: If label-map JSON is invalid
        """
        label_map = None
        if getattr(args, "label_map", None):
            try:
                label_map = json.loads(args.label_map)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON for label-map: {e}")

        return cls(
            label_map=label_map,
            source_prefix=getattr(args, "source_prefix", cls.SOURCE_PREFIX),
            destination_prefix=getattr(args, "dest_prefix", cls.DEST_PREFIX),
            s3_bucket=args.s3_bucket,
            verbose=args.verbose,
            log_file=args.log_file,
            dry_run=args.dry_run,
            keep_local=getattr(args, "keep_local", False),
        )

    # ------------------------------------------------------------------ #
    # Session discovery
    # ------------------------------------------------------------------ #

    def find_sessions(self) -> List[str]:
        """
        Find sessions with both event and window zarr stores.
        
        Returns:
            List of session IDs with both event and window data
        """
        # Find sessions with event zarr stores
        event_sessions: Set[str] = set()
        pattern = "_events.zarr/zarr.json"

        resp = self.s3.list_objects_v2(Bucket=self.s3_bucket, Prefix=self.source_prefix)
        for obj in resp.get("Contents", []):
            key = obj["Key"]
            if key.endswith(pattern):
                sid = key.split("/")[-2].replace("_events.zarr", "")
                event_sessions.add(sid)

        self.logger.info(f"Found {len(event_sessions)} sessions with event zarr stores")
        
        # Find sessions with window zarr stores
        window_sessions: Set[str] = set()
        window_prefix = "processed/windows/"
        
        resp = self.s3.list_objects_v2(Bucket=self.s3_bucket, Prefix=window_prefix)
        for obj in resp.get("Contents", []):
            key = obj["Key"]
            if key.endswith("_windowed.zarr/zarr.json"):
                sid = key.split("/")[-2].replace("_windowed.zarr", "")
                window_sessions.add(sid)
                
        self.logger.info(f"Found {len(window_sessions)} sessions with window zarr stores")
        
        # Find sessions with both stores
        common_sessions = event_sessions.intersection(window_sessions)
        self.logger.info(f"Found {len(common_sessions)} sessions with both event and window data")
        
        return sorted(common_sessions)

    # ------------------------------------------------------------------ #
    # Core processing
    # ------------------------------------------------------------------ #

    def process_session(self, session: Session) -> Dict[str, Any]:
        """
        Process a single session by extracting eye neural windows.
        
        Args:
            session: Session object containing session information
            
        Returns:
            Dictionary with processing results and metadata
        """
        sid = session.session_id
        self.logger.info(f"Processing session {sid}")

        # Paths
        event_key = (
            f"{self.source_prefix}{sid}"
            if sid.endswith("_events.zarr")
            else f"{self.source_prefix}{sid}_events.zarr"
        )
        clean_sid = sid.replace("_events.zarr", "")
        window_key = f"processed/windows/{clean_sid}_windowed.zarr"

        # Existence checks
        try:
            self.s3.head_object(Bucket=self.s3_bucket, Key=f"{event_key}/zarr.json")
        except self.s3.exceptions.ClientError:
            self.logger.warning(f"No events zarr for {sid}")
            return {"status": "skipped", "metadata": {"reason": "missing_event_zarr", "session_id": sid}}

        try:
            self.s3.head_object(Bucket=self.s3_bucket, Key=f"{window_key}/zarr.json")
        except self.s3.exceptions.ClientError:
            self.logger.warning(f"No window zarr for {sid}")
            return {"status": "skipped", "metadata": {"reason": "missing_window_zarr", "session_id": sid}}

        # Open stores
        try:
            eg = open_zarr_safely(f"s3://{self.s3_bucket}/{event_key}", logger=self.logger)
            wg = open_zarr_safely(f"s3://{self.s3_bucket}/{window_key}", logger=self.logger)
        except Exception as e:
            self.logger.error(f"Unable to open zarr stores for {sid}: {e}")
            return {
                "status": "failed", 
                "error_details": str(e),
                "metadata": {"session_id": sid, "error_type": "zarr_open_error"}
            }

        # Extract data
        try:
            result = extract_elements_neural_windows(
                event_group=eg,
                window_group=wg,
                session_id=sid,
                filter_kwargs={"task_type": "eye"},
                label_map=self.label_map,
                logger=self.logger,
            )
        except Exception as e:
            self.logger.error(f"Extraction failed for {sid}: {e}", exc_info=True)
            return {
                "status": "failed", 
                "error_details": str(e),
                "metadata": {"session_id": sid, "error_type": "extraction_error"}
            }

        if result is None:
            return {"status": "skipped", "metadata": {"reason": "no_eye_elements", "session_id": sid}}

        # Calculate statistics
        window_count = len(result.get("labels", []))
        element_count = len(np.unique(result.get("element_ids", [])))
        
        # Count windows per label
        label_counts = {}
        if "labels" in result:
            unique_labels, counts = np.unique(result["labels"], return_counts=True)
            for lbl, count in zip(unique_labels, counts):
                label_name = next((k for k, v in self.label_map.items() if v == lbl), f"label_{lbl}")
                label_counts[label_name] = int(count)
        
        self.logger.info(f"Session {sid}: extracted {window_count} windows across {element_count} eye elements")

        # Save results if not dry run
        out_key = f"{self.destination_prefix}eye_neural.zarr"
        overwrite = getattr(self, "include_processed", False)

        if not self.dry_run:
            root, sessions_group, _ = init_or_open_result_store(
                s3_client=self.s3,
                s3_bucket=self.s3_bucket,
                result_key=out_key,
                query_name="eye_neural",
                label_map=self.label_map,
                logger=self.logger,
            )
            save_session_to_subgroup(sessions_group, sid, result, logger=self.logger, overwrite=overwrite)
            write_session_result(root, result, sid, logger=self.logger)

        return {
            "status": "success",
            "metadata": {
                "session_id": sid,
                "window_count": window_count,
                "element_count": element_count,
                "label_counts": label_counts,
                "query_type": "eye_neural",
            },
            "zarr_stores": [out_key],
        }


# ---------------------------------------------------------------------- #
# CLI entry-point
# ---------------------------------------------------------------------- #

if __name__ == "__main__":
    EyeNeuralTransform.run_from_command_line()