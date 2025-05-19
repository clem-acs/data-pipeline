import zarr, numpy as np, collections, logging

logging.basicConfig(level=logging.INFO)
root = zarr.open(
    "s3://conduit-data-dev/processed/queries/eye_neural.zarr",
    mode="r", storage_options={"anon": False}
)

for sid, sg in root["sessions"].groups():
    eeg = sg["eeg"]
    labels = sg["labels"][:]
    kept   = []
    for i in range(len(labels)):
        if eeg[i].shape == (21, 105):      # same test as QueryDataset
            kept.append(labels[i])
    c = collections.Counter(kept)
    print(f"{sid:35s}  kept={sum(c.values()):3d}  {dict(c)}")
