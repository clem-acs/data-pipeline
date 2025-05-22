"""
T4C Neuro-Lang Transform – version 0.0

Single-query transform that links language tokens to simultaneous neural
windows.  
The result is aggregated into a consolidated Zarr v3 store with a path that
includes the language group, tokenizer, and pre-window seconds:

    processed/queries/neuro_lang_{lang_group}_{tokenizer}_{pre_window_seconds}s.zarr
"""

import json
import logging
from typing import Any, Dict, List, Optional, Set

import numpy as np
import zarr

from base_transform import BaseTransform, Session
from transforms.query_helpers import (
    # generic helpers
    open_zarr_safely,
    window_indices_for_range,
    extract_full_windows,
    extract_elements_data,
    # language helpers
    extract_lang_tokens,
    find_element_for_timestamp,
    # store helpers
    init_or_open_result_store,
    save_session_to_subgroup,
    write_session_result,
    bytesify,
)

logger = logging.getLogger(__name__)


class NeuroLangTransform(BaseTransform):
    """
    Extract neural windows that occur around language tokens.

    • finds sessions that have:
        – *_events.zarr          (elements)
        – *_{tokenizer}_lang.zarr (language)
        – *_windowed.zarr         (neural windows)
    • for every token in the chosen language group, grabs all windows whose
      timestamps lie in `[token_start – pre_window_seconds , token_end]`
    • saves a session subgroup inside a parameterized path:
      `processed/queries/neuro_lang_{lang_group}_{tokenizer}_{pre_window_seconds}s.zarr`
    """

    SOURCE_PREFIX = "processed/event/"
    DEST_PREFIX = "processed/queries/"

    # ------------------------------------------------------------------ #
    # constructor / CLI
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        lang_group: str = "W",
        tokenizer: str = "gpt2",
        pre_window_seconds: float = 10.0,
        **kwargs,
    ):
        """
        Initialize the Neuro-Lang transform.
        
        Args:
            lang_group: Language group to process (L, W, R, S, W_corrected)
            tokenizer: Tokenizer used in language transform (e.g. gpt2)
            pre_window_seconds: Seconds to include before token start
            **kwargs: Additional arguments for BaseTransform
        """
        self.lang_group = lang_group
        self.tokenizer = tokenizer
        self.pre_window_seconds = pre_window_seconds

        # Build a transform_id that includes all parameters
        transform_id = kwargs.pop(
            "transform_id", 
            f"t4C_neuro_lang_v0_{lang_group}_{tokenizer}_{pre_window_seconds}s"
        )
        script_id = kwargs.pop("script_id", "4C")
        script_name = kwargs.pop("script_name", "neuro_lang")
        script_version = kwargs.pop("script_version", "v0")

        super().__init__(
            transform_id=transform_id,
            script_id=script_id,
            script_name=script_name,
            script_version=script_version,
            **kwargs,
        )

        self.logger.info(
            f"NeuroLangTransform initialized "
            f"(lang_group={self.lang_group}, tokenizer={self.tokenizer}, "
            f"pre_window_seconds={self.pre_window_seconds})"
        )

    @classmethod
    def add_subclass_arguments(cls, parser) -> None:
        """
        Add transform-specific command-line arguments.
        
        Args:
            parser: ArgumentParser instance
        """
        parser.add_argument(
            "--lang-group",
            choices=["L", "R", "W", "S", "W_corrected"],
            default="W",
            help="Language group to extract (default: W)",
        )
        parser.add_argument(
            "--tokenizer",
            type=str,
            default="gpt2",
            help="Tokenizer string used in the language transform (default: gpt2)",
        )
        parser.add_argument(
            "--pre-window-seconds",
            type=float,
            default=10.0,
            help="Seconds of neural data to include *before* each token (default: 10)",
        )

    @classmethod
    def from_args(cls, args) -> "NeuroLangTransform":
        """
        Create transform instance from command-line arguments.
        
        Args:
            args: Parsed command-line arguments
            
        Returns:
            Configured NeuroLangTransform instance
        """
        return cls(
            lang_group=getattr(args, "lang_group", "W"),
            tokenizer=getattr(args, "tokenizer", "gpt2"),
            pre_window_seconds=getattr(args, "pre_window_seconds", 10.0),
            source_prefix=getattr(args, "source_prefix", cls.SOURCE_PREFIX),
            destination_prefix=getattr(args, "dest_prefix", cls.DEST_PREFIX),
            s3_bucket=args.s3_bucket,
            verbose=args.verbose,
            log_file=args.log_file,
            dry_run=args.dry_run,
            keep_local=getattr(args, "keep_local", False),
        )

    # ------------------------------------------------------------------ #
    # session discovery
    # ------------------------------------------------------------------ #

    def find_sessions(self) -> List[str]:
        """
        Return sessions that have all three required stores:
        event + language (with correct tokenizer) + window.
        
        Returns:
            List of session IDs with all required data
        """
        bucket = self.s3_bucket

        # 1 events
        event_sessions: Set[str] = set()
        resp = self.s3.list_objects_v2(Bucket=bucket, Prefix=self.SOURCE_PREFIX)
        for obj in resp.get("Contents", []):
            key = obj["Key"]
            if key.endswith("_events.zarr/zarr.json"):
                event_sessions.add(key.split("/")[-2].replace("_events.zarr", ""))

        # 2 language with specific tokenizer
        lang_sessions: Set[str] = set()
        lang_prefix = "processed/lang/"
        resp = self.s3.list_objects_v2(Bucket=bucket, Prefix=lang_prefix)
        for obj in resp.get("Contents", []):
            key = obj["Key"]
            if key.endswith("_lang.zarr/zarr.json") and f"_{self.tokenizer}_lang.zarr" in key:
                sid = key.split("/")[-2].replace(f"_{self.tokenizer}_lang.zarr", "")
                lang_sessions.add(sid)

        # 3 window stores
        win_sessions: Set[str] = set()
        win_prefix = "processed/windows/"
        resp = self.s3.list_objects_v2(Bucket=bucket, Prefix=win_prefix)
        for obj in resp.get("Contents", []):
            key = obj["Key"]
            if key.endswith("_windowed.zarr/zarr.json"):
                win_sessions.add(key.split("/")[-2].replace("_windowed.zarr", ""))

        common = event_sessions & lang_sessions & win_sessions
        self.logger.info(
            f"Found {len(common)} sessions with event, "
            f"{self.tokenizer} language and window stores"
        )
        return sorted(common)

    # ------------------------------------------------------------------ #
    # per-session processing
    # ------------------------------------------------------------------ #

    def process_session(self, session: Session) -> Dict[str, Any]:
        """
        Process a single session by extracting neural windows for language tokens.
        
        Args:
            session: Session object containing session information
            
        Returns:
            Dictionary with processing results and metadata
        """
        sid = session.session_id
        self.logger.info(f"Processing session {sid}")

        bucket = self.s3_bucket
        clean_sid = sid.replace("_events.zarr", "")

        event_key = f"{self.SOURCE_PREFIX}{clean_sid}_events.zarr"
        lang_key = f"processed/lang/{clean_sid}_{self.tokenizer}_lang.zarr"
        window_key = f"processed/windows/{clean_sid}_windowed.zarr"

        # head-checks
        for k, reason in [
            (event_key, "no_event_zarr"),
            (lang_key, "no_lang_zarr"),
            (window_key, "no_window_zarr"),
        ]:
            try:
                self.s3.head_object(Bucket=bucket, Key=f"{k}/zarr.json")
            except self.s3.exceptions.ClientError:
                return {"status": "skipped", "metadata": {"reason": reason, "session_id": sid}}

        # open stores
        try:
            eg = open_zarr_safely(f"s3://{bucket}/{event_key}", logger=self.logger)
            lg = open_zarr_safely(f"s3://{bucket}/{lang_key}", logger=self.logger)
            wg = open_zarr_safely(f"s3://{bucket}/{window_key}", logger=self.logger)
        except Exception as e:
            self.logger.error(f"Zarr open failed for {sid}: {e}", exc_info=True)
            return {"status": "failed", "error_details": str(e), "metadata": {"session_id": sid}}

        # ------------------------------------------------------------------
        # token → window linking
        # ------------------------------------------------------------------
        elements = extract_elements_data(eg, sid, logger=self.logger)
        if elements is None:
            return {"status": "skipped", "metadata": {"reason": "no_elements", "session_id": sid}}

        lang_data = extract_lang_tokens(lg, self.lang_group)
        if lang_data is None or "tokens" not in lang_data or not lang_data["tokens"]:
            return {"status": "skipped", "metadata": {"reason": "no_tokens", "session_id": sid}}

        window_times = wg["time"][:]

        tokens = lang_data["tokens"]
        order = sorted(range(len(tokens)), key=lambda i: tokens[i]["start_timestamp"])

        # accumulators
        tok_text, tok_ids, t_start, t_end, t_special, t_eids = [], [], [], [], [], []
        token_idx_map, window_idx_map = [], []
        windows_cache: Dict[int, Dict[str, np.ndarray]] = {}

        for idx in order:
            tk = tokens[idx]
            s_ts = tk.get("start_timestamp", 0)
            e_ts = tk.get("end_timestamp", s_ts)
            eid = find_element_for_timestamp(elements, s_ts)
            if eid is None:
                continue

            win_start = max(0, s_ts - int(self.pre_window_seconds * 1000))
            hits = window_indices_for_range(window_times, win_start, e_ts)
            if not hits.size:
                continue

            # token arrays
            tok_text.append(tk.get("token", ""))
            tok_ids.append(tk.get("token_id", 0))
            t_start.append(s_ts)
            t_end.append(e_ts)
            t_special.append(tk.get("special_token", False))
            t_eids.append(eid)

            this_tok_idx = len(tok_text) - 1
            for w_idx in hits:
                token_idx_map.append(this_tok_idx)
                window_idx_map.append(int(w_idx))
                if w_idx not in windows_cache:
                    wd = extract_full_windows(wg, np.array([w_idx]), logger=self.logger)
                    windows_cache[int(w_idx)] = {
                        k: v[0] if isinstance(v, np.ndarray) and v.shape[0] == 1 else v
                        for k, v in wd.items()
                    }

        if not tok_text:
            return {"status": "skipped", "metadata": {"reason": "no_matches", "session_id": sid}}

        # vectorise window cache
        ordered_w_keys = sorted(windows_cache.keys())
        window_arrays: Dict[str, np.ndarray] = {}
        first = next(iter(windows_cache.values()))
        for key in first:
            vals = [windows_cache[i][key] for i in ordered_w_keys]
            window_arrays[key] = (
                np.stack(vals) if all(isinstance(v, np.ndarray) for v in vals) else np.array(vals)
            )

        # Use bytesify on arrays that might contain strings
        for key in list(window_arrays.keys()):
            if window_arrays[key].dtype.kind in ('O', 'U'):
                window_arrays[key] = bytesify(window_arrays[key])

        w_idx_remap = {old: i for i, old in enumerate(ordered_w_keys)}
        remapped_w_idx = [w_idx_remap[i] for i in window_idx_map]

        result = {
            "session_id": sid,
            "tokenizer": self.tokenizer,
            "lang_group": self.lang_group,
            "pre_window_seconds": self.pre_window_seconds,
            "tokens": {
                "text": np.array(tok_text),
                "ids": np.array(tok_ids),
                "start_times": np.array(t_start),
                "end_times": np.array(t_end),
                "special": np.array(t_special),
                "element_ids": np.array(t_eids),
            },
            "windows": window_arrays,
            "token_window_map": {
                "token_idx": np.array(token_idx_map),
                "window_idx": np.array(remapped_w_idx),
            },
        }

        # Print debug information
        token_count = len(tok_text)
        window_count = len(ordered_w_keys)
        mapping_count = len(token_idx_map)
        
        self.logger.info(f"Session {sid}: extracted {token_count} tokens with {window_count} windows, {mapping_count} mappings")
        
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f"Result keys: {list(result.keys())}")
            self.logger.debug(f"Token count: {token_count}")
            self.logger.debug(f"Window count: {window_count}")
            self.logger.debug(f"Token-window mapping count: {mapping_count}")
            
            if window_arrays:
                self.logger.debug(f"Window array keys: {list(window_arrays.keys())}")
                for k, v in window_arrays.items():
                    if isinstance(v, np.ndarray):
                        self.logger.debug(f"  {k}: shape={v.shape}, dtype={v.dtype}")

        # ------------------------------------------------------------------
        # save
        # ------------------------------------------------------------------
        
        # Build parameterized output path
        out_key = f"{self.DEST_PREFIX}neuro_lang_{self.lang_group}_{self.tokenizer}_{self.pre_window_seconds}s.zarr"
        query_name = f"neuro_lang_{self.lang_group}_{self.tokenizer}"
        overwrite = getattr(self, "include_processed", False)

        if not self.dry_run:
            root, sess_grp, _ = init_or_open_result_store(
                s3_client=self.s3,
                s3_bucket=bucket,
                result_key=out_key,
                query_name=query_name,
                label_map=None,
                logger=self.logger,
            )
            save_session_to_subgroup(sess_grp, sid, result, logger=self.logger, overwrite=overwrite)
            write_session_result(root, result, sid, logger=self.logger)

        return {
            "status": "success",
            "metadata": {
                "session_id": sid,
                "token_count": token_count,
                "window_count": window_count,
                "mapping_count": mapping_count,
                "query_type": "neuro_lang",
                "lang_group": self.lang_group,
                "tokenizer": self.tokenizer,
                "pre_window_seconds": self.pre_window_seconds,
            },
            "zarr_stores": [out_key],
        }


# ---------------------------------------------------------------------- #
# CLI entry-point
# ---------------------------------------------------------------------- #

if __name__ == "__main__":
    NeuroLangTransform.run_from_command_line()