import zarr, numpy as np
from collections import Counter

ZARR_URI = "s3://conduit-data-dev/processed/queries/eye_neural.zarr"
LABEL_MAP = {"closed": 0, "open": 1, "intro": 2, "unknown": 3}
INV_LABEL = {v: k for k, v in LABEL_MAP.items()}

root   = zarr.open_group(ZARR_URI, mode="r", storage_options={"anon": False})
sess   = next(iter(root["sessions"]))
sg     = root["sessions"][sess]

print(f"\n=== Session: {sess} ===")
for key in ("eeg", "fnirs", "time", "labels", "element_ids"):
    if key in sg:
        print(f"{key:<12s} shape={sg[key].shape}, dtype={sg[key].dtype}")
print()

# ---- label stats ----
labels = sg["labels"][:]
if labels.dtype.kind in ("S", "a"):
    labels = labels.astype(str)                 # string labels
cnt, total = Counter(labels), len(labels)

print(f"Total windows: {total}\nLabel distribution:")
for lbl_id, n in cnt.items():
    # translate if numeric, else keep as string
    name = INV_LABEL.get(lbl_id, str(lbl_id))
    print(f"  {name:<8s} ({lbl_id}): {n:4d}  ({n*100/total:4.1f}%)")

# ---- quick peek at first 3 windows ----
times = sg["time"][:3]
eids  = sg["element_ids"][:3].astype(str)
print("\nFirst 3 windows:")
for i in range(3):
    lbl = labels[i]
    lbl_name = INV_LABEL.get(lbl, str(lbl))
    print(f"  idx {i:<2d} ts={times[i]}  eid={eids[i]}  label={lbl_name}")

# ---- extra sanity check ----
N = total
assert all(sg[a].shape[0] == N for a in ("eeg", "fnirs", "time")), \
       "window axis length mismatch!"
print("\nDataset looks consistent ")
