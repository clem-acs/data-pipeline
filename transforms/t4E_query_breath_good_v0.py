"""
T4E Breath-Good Transform – version 0.0

Extract neural windows that fall inside the middle X-percent of every
*breath-task* element whose duration exceeds S seconds.

Defaults
--------
• --min-duration-seconds (-S) : 10 s
• --middle-percent (-X)       : 0.5   ⇢ middle 50 %

Output
------
processed/queries/breath_<S>_<X>.zarr
    ├─ root.attrs.query_name            = "breath_good"
    ├─ root.attrs.min_duration_seconds  = S
    └─ root.attrs.middle_percent        = X
Each qualifying session is stored under /sessions/<session_id>/…
"""

# --------------------------------------------------------------------- #
# Std-lib / third-party
# --------------------------------------------------------------------- #
import json
import logging
from typing import Any, Dict, List, Optional, Set

import numpy as np
import zarr

# --------------------------------------------------------------------- #
# Pipeline imports
# --------------------------------------------------------------------- #
from base_transform import BaseTransform, Session
from transforms.query_helpers import (
    open_zarr_safely,
    bytesify,
    window_indices_for_range,
    extract_full_windows,
    extract_elements_data,
    build_labels,
    build_eids,
    extract_elements_neural_windows,  # Added the enhanced function
    init_or_open_result_store,
    save_session_to_subgroup,
    write_session_result,
)

# --------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------- #
DEFAULT_BREATH_LABEL_MAP = {
    "hold":    0,   # analogous to "close"
    "normal":  1,   # analogous to "open"
    "intro":   2,
    "unknown": 3,
}
logger = logging.getLogger(__name__)


# ===================================================================== #
#  Transform
# ===================================================================== #
class BreathGoodTransform(BaseTransform):
    """
    Same logic as Eye-Good (t4D) but for *breath* elements.
    """

    SOURCE_PREFIX = "processed/event/"
    DEST_PREFIX   = "processed/queries/"

    # ------------------------------------------------------------------ #
    # ctor / CLI
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        min_duration_seconds: float = 10.0,
        middle_percent: float       = 0.5,
        label_map: Optional[Dict[str, int]] = None,
        **kwargs,
    ):
        self.min_duration_seconds = float(min_duration_seconds)
        self.middle_percent       = float(middle_percent)
        if not (0.0 < self.middle_percent <= 1.0):
            raise ValueError("--middle-percent must be in (0, 1]")

        self.label_map = label_map or DEFAULT_BREATH_LABEL_MAP

        transform_id = kwargs.pop(
            "transform_id",
            f"t4E_breath_good_v0_{int(self.min_duration_seconds)}s_{self.middle_percent}",
        )
        script_id      = kwargs.pop("script_id", "4E")
        script_name    = kwargs.pop("script_name", "breath_good")
        script_version = kwargs.pop("script_version", "v0")

        super().__init__(
            transform_id   = transform_id,
            script_id      = script_id,
            script_name    = script_name,
            script_version = script_version,
            **kwargs,
        )

        self.logger.info(
            f"BreathGoodTransform initialised "
            f"(min_duration_seconds={self.min_duration_seconds}, "
            f"middle_percent={self.middle_percent}, "
            f"label_map={self.label_map})"
        )

    # ----------------------------- CLI hooks -------------------------- #
    @classmethod
    def add_subclass_arguments(cls, parser) -> None:
        parser.add_argument(
            "--min-duration-seconds",
            type=float,
            default=10.0,
            help="Only breath elements longer than this many seconds "
                 "are kept (default: 10).",
        )
        parser.add_argument(
            "--middle-percent",
            type=float,
            default=0.5,
            help="Fraction (0 < p ≤ 1) of each qualifying element's duration "
                 "to keep, centred in the middle (default: 0.5).",
        )
        parser.add_argument(
            "--label-map",
            type=str,
            help="Optional JSON dict mapping element-id substrings → label "
                 "e.g. '{\"hold\":0,\"normal\":1}'.",
        )

    @classmethod
    def from_args(cls, args) -> "BreathGoodTransform":
        label_map = None
        if getattr(args, "label_map", None):
            try:
                label_map = json.loads(args.label_map)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON for --label-map: {e}")

        return cls(
            min_duration_seconds = getattr(args, "min_duration_seconds", 10.0),
            middle_percent       = getattr(args, "middle_percent", 0.5),
            label_map            = label_map,
            source_prefix        = getattr(args, "source_prefix", cls.SOURCE_PREFIX),
            destination_prefix   = getattr(args, "dest_prefix",    cls.DEST_PREFIX),
            s3_bucket            = args.s3_bucket,
            verbose              = args.verbose,
            log_file             = args.log_file,
            dry_run              = args.dry_run,
            keep_local           = getattr(args, "keep_local", False),
        )

    # ------------------------------------------------------------------ #
    # session discovery
    # ------------------------------------------------------------------ #
    def find_sessions(self) -> List[str]:
        """
        Sessions must have *_events.zarr* **and** *_windowed.zarr*.
        """
        bkt = self.s3_bucket
        ev_sessions, win_sessions = set(), set()

        # events
        resp = self.s3.list_objects_v2(Bucket=bkt, Prefix=self.SOURCE_PREFIX)
        for o in resp.get("Contents", []):
            k = o["Key"]
            if k.endswith("_events.zarr/zarr.json"):
                ev_sessions.add(k.split("/")[-2].replace("_events.zarr", ""))

        # windows
        resp = self.s3.list_objects_v2(Bucket=bkt, Prefix="processed/windows/")
        for o in resp.get("Contents", []):
            k = o["Key"]
            if k.endswith("_windowed.zarr/zarr.json"):
                win_sessions.add(k.split("/")[-2].replace("_windowed.zarr", ""))

        sessions = sorted(ev_sessions & win_sessions)
        self.logger.info(f"Found {len(sessions)} sessions with event + window data")
        return sessions

    # ------------------------------------------------------------------ #
    # per-session processing
    # ------------------------------------------------------------------ #
    def process_session(self, session: Session) -> Dict[str, Any]:
        sid = session.session_id
        bkt = self.s3_bucket
        self.logger.info(f"Processing session {sid}")

        clean_sid = sid.replace("_events.zarr", "")
        event_key = f"{self.SOURCE_PREFIX}{clean_sid}_events.zarr"
        window_key = f"processed/windows/{clean_sid}_windowed.zarr"

        # Quick existence checks
        for key, reason in [
            (event_key, "no_event_zarr"),
            (window_key, "no_window_zarr"),
        ]:
            try:
                self.s3.head_object(Bucket=bkt, Key=f"{key}/zarr.json")
            except self.s3.exceptions.ClientError:
                return {"status": "skipped",
                        "metadata": {"reason": reason, "session_id": sid}}

        # Open zarrs
        try:
            eg = open_zarr_safely(f"s3://{bkt}/{event_key}", logger=self.logger)
            wg = open_zarr_safely(f"s3://{bkt}/{window_key}", logger=self.logger)
        except Exception as e:
            self.logger.error(f"Zarr open failed: {e}", exc_info=True)
            return {"status": "failed", "error_details": str(e),
                    "metadata": {"session_id": sid}}

        # Use enhanced helper function to extract data
        windows = extract_elements_neural_windows(
            event_group=eg,
            window_group=wg,
            session_id=sid,
            filter_kwargs={"task_type": "breath"},
            label_map=self.label_map,
            min_duration_seconds=self.min_duration_seconds,
            middle_percent=self.middle_percent,
            logger=self.logger
        )

        if windows is None:
            return {"status": "skipped",
                    "metadata": {"reason": "no_qualifying_windows", "session_id": sid}}

        # Debug print for final label distribution
        unique_labels, counts = np.unique(windows["labels"], return_counts=True)
        print(f"DEBUG: Final label distribution: {dict(zip(unique_labels, counts))}")
        print(f"DEBUG: Label map used: {self.label_map}")

        # stats
        lbls = windows.get("labels", np.empty(0))
        w_cnt = len(windows.get("labels", []))
        e_cnt = len(np.unique(windows.get("element_ids", [])))

        label_counts: Dict[str, int] = {}
        if lbls.size:
            uniq, cts = np.unique(lbls, return_counts=True)
            for lbl, ct in zip(uniq, cts):
                name = next((k for k, v in self.label_map.items() if v == lbl),
                            f"label_{lbl}")
                label_counts[name] = int(ct)

        self.logger.info(f"{sid}: {w_cnt} windows from {e_cnt} breath elements")

        # --------------------------- save ----------------------------- #
        out_key = (f"{self.destination_prefix}"
                   f"breath_{int(self.min_duration_seconds)}_"
                   f"{self.middle_percent}.zarr")
        overwrite = getattr(self, "include_processed", False)

        if not self.dry_run:
            root, sess_grp, _ = init_or_open_result_store(
                s3_client  = self.s3,
                s3_bucket  = bkt,
                result_key = out_key,
                query_name = "breath_good",
                label_map  = self.label_map,
                logger     = self.logger,
            )
            root.attrs["min_duration_seconds"] = self.min_duration_seconds
            root.attrs["middle_percent"]       = self.middle_percent
            save_session_to_subgroup(
                sess_grp, sid, windows, logger=self.logger, overwrite=overwrite
            )
            write_session_result(root, windows, sid, logger=self.logger)

        return {
            "status": "success",
            "metadata": {
                "session_id":    sid,
                "window_count":  w_cnt,
                "element_count": e_cnt,
                "label_counts":  label_counts,
                "query_type":    "breath_good",
                "min_duration_seconds": self.min_duration_seconds,
                "middle_percent":       self.middle_percent,
            },
            "zarr_stores": [out_key],
        }


# --------------------------------------------------------------------- #
# CLI entry-point
# --------------------------------------------------------------------- #
if __name__ == "__main__":
    BreathGoodTransform.run_from_command_line()