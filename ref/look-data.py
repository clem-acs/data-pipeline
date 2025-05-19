import zarr, numpy as np
from collections import defaultdict, Counter

# ---------- config ----------
ZARR_URI  = "s3://conduit-data-dev/processed/queries/eye_neural.zarr"
LABEL_MAP = {"close": 0, "open": 1, "intro": 2, "unknown": 3}
INV_LABEL = {v: k for k, v in LABEL_MAP.items()}
N_EXAMPLES = 5
# ----------------------------

root = zarr.open_group(ZARR_URI, mode="r",
                       storage_options={"anon": False})

all_labels, all_eids, all_times = [], [], []

sessions = root["sessions"]                         # parent group

for sid in sessions.group_keys():                  # <-- here
    sg = sessions[sid]                             # session subgroup
    all_labels.append(sg["labels"][:])
    all_eids  .append(sg["element_ids"][:].astype(str))
    all_times .append(sg["time"][:])

labels      = np.concatenate(all_labels)
element_ids = np.concatenate(all_eids)
times       = np.concatenate(all_times)

# byte-to-str if needed
if labels.dtype.kind in ("S", "a"):
    labels = labels.astype(str)

cnt = Counter(labels)
total = len(labels)

print(f"\nTotal windows: {total}")
print(f"Label breakdown (showing {N_EXAMPLES} unique element_ids each)\n")

examples  = defaultdict(set)
time_span = defaultdict(lambda: [np.inf, -np.inf])

for lbl, eid, ts in zip(labels, element_ids, times):
    examples[lbl].add(eid)
    t0, t1 = time_span[lbl]
    time_span[lbl] = [min(t0, ts), max(t1, ts)]

for lbl, n in cnt.items():
    name  = INV_LABEL.get(lbl, str(lbl))
    eids  = sorted(examples[lbl])[:N_EXAMPLES]
    t0, t1 = time_span[lbl]
    print(f"{name:<8s} ({lbl}): {n:4d}  ({n*100/total:4.1f}%)")
    print(f"  first ts: {t0:.0f}   last ts: {t1:.0f}")
    print(f"  sample eids: {', '.join(eids)}\n")
