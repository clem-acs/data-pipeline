"""
t4G Eng-Fin – v0

Middle-X% neural windows from *language* elements ≥ S seconds
→ processed/queries/eng_fin_<S>_<X>.zarr
"""

import json, logging, numpy as np
from typing import Any, Dict, List, Optional
from base_transform import BaseTransform, Session
from transforms.query_helpers import (
    open_zarr_safely, extract_elements_neural_windows,
    init_or_open_result_store, save_session_to_subgroup, write_session_result,
)

LABELS = {"en": 0, "fi": 1, "intro": 2, "unknown": 3}
log = logging.getLogger(__name__)


class EngFinTransform(BaseTransform):
    """Extract neural windows from middle X-percent of language task elements."""
    SOURCE_PREFIX = "processed/event/"
    DEST_PREFIX   = "processed/queries/"

    def __init__(self, min_duration_seconds=9.0, middle_percent=0.5,
                 label_map: Optional[Dict[str, int]] = None, **kw):
        if not 0 < middle_percent <= 1: raise ValueError("--middle-percent out of range")
        self.S, self.X = float(min_duration_seconds), float(middle_percent)
        self.label_map = label_map or LABELS
        super().__init__(transform_id=f"t4G_eng_fin_v0_{int(self.S)}s_{self.X}", 
                         script_id="4G", script_name="eng_fin", script_version="v0", **kw)

    # ---------- CLI ----------
    @classmethod
    def add_subclass_arguments(cls, p):
        p.add_argument("--min-duration-seconds", type=float, default=11)
        p.add_argument("--middle-percent",      type=float, default=0.65)
        p.add_argument("--label-map", type=str)

    @classmethod
    def from_args(cls, a):
        lm = json.loads(a.label_map) if a.label_map else None
        return cls(a.min_duration_seconds, a.middle_percent, lm,
                   source_prefix=a.source_prefix, destination_prefix=a.dest_prefix,
                   s3_bucket=a.s3_bucket, verbose=a.verbose, log_file=a.log_file,
                   dry_run=a.dry_run, keep_local=a.keep_local)

    # ---------- discovery ----------
    def find_sessions(self) -> List[str]:
        b = self.s3_bucket; ev, wn = set(), set()
        for o in self.s3.list_objects_v2(Bucket=b, Prefix=self.SOURCE_PREFIX).get("Contents", []):
            if o["Key"].endswith("_events.zarr/zarr.json"):
                ev.add(o["Key"].split("/")[-2].replace("_events.zarr", ""))
        for o in self.s3.list_objects_v2(Bucket=b, Prefix="processed/windows/").get("Contents", []):
            if o["Key"].endswith("_windowed.zarr/zarr.json"):
                wn.add(o["Key"].split("/")[-2].replace("_windowed.zarr", ""))
        return sorted(ev & wn)

    # ---------- per-session ----------
    def process_session(self, s: Session) -> Dict[str, Any]:
        sid, b = s.session_id, self.s3_bucket
        ek=f"{self.SOURCE_PREFIX}{sid.replace('_events.zarr', '')}_events.zarr" 
        wk=f"processed/windows/{sid.replace('_events.zarr', '')}_windowed.zarr"
        for k,r in [(ek,"no_event"),(wk,"no_window")]:
            try: self.s3.head_object(Bucket=b, Key=f"{k}/zarr.json")
            except self.s3.exceptions.ClientError: return {"status":"skipped","metadata":{"reason":r,"session_id":sid}}
        try:
            eg=open_zarr_safely(f"s3://{b}/{ek}",logger=log); wg=open_zarr_safely(f"s3://{b}/{wk}",logger=log)
        except Exception as e:
            return {"status":"failed","error_details":str(e),"metadata":{"session_id":sid}}

        win = extract_elements_neural_windows(eg, wg, sid,
                filter_kwargs={"task_type":"language"}, label_map=self.label_map,
                min_duration_seconds=self.S, middle_percent=self.X, logger=log)
        if win is None:
            return {"status":"skipped","metadata":{"reason":"no_windows","session_id":sid}}

        u,c=np.unique(win["labels"],return_counts=True)
        print(f"DEBUG: Final label distribution: {dict(zip(u,c))}")
        print(f"DEBUG: Label map used: {self.label_map}")

        # Calculate label counts
        label_counts = {}
        if len(u) > 0:
            for lbl, ct in zip(u, c):
                name = next((k for k, v in self.label_map.items() if v == lbl), f"label_{lbl}")
                label_counts[name] = int(ct)

        out=f"{self.DEST_PREFIX}eng_fin_{int(self.S)}_{self.X}.zarr"
        if not self.dry_run:
            root,sgrp,_=init_or_open_result_store(self.s3,b,out,"eng_fin",self.label_map,log)
            root.attrs.update(min_duration_seconds=self.S,middle_percent=self.X)
            save_session_to_subgroup(sgrp,sid,win,logger=log,overwrite=getattr(self,"include_processed",False))
            write_session_result(root,win,sid,logger=log)

        # Enhanced metadata
        return {
            "status":"success",
            "metadata":{
                "session_id": sid,
                "window_count": len(win.get("labels", [])),
                "element_count": len(np.unique(win.get("element_ids", []))),
                "label_counts": label_counts,
                "query_type": "eng_fin",
                "min_duration_seconds": self.S,
                "middle_percent": self.X,
            },
            "zarr_stores":[out]
        }


if __name__ == "__main__":
    EngFinTransform.run_from_command_line()