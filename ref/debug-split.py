#!/usr/bin/env python
import argparse, torch
from utils import create_data_loaders

p = argparse.ArgumentParser()
p.add_argument("query")
p.add_argument("--bucket", default="conduit-data-dev")
args = p.parse_args()

tr, va, te = create_data_loaders(
        f"s3://{args.bucket}/processed/queries/{args.query}.zarr",
        modalities=["eeg"], split_by_session=True, shuffle=False, num_workers=0)

for name, dl in [("train", tr), ("val", va), ("test", te)]:
    if not dl: print(f"{name}: 0 windows"); continue
    ys = torch.cat([y for _,y in dl])
    u,c = ys.unique(return_counts=True)
    print(f"{name:5s}: {len(ys):5d} windows | {dict(zip(u.tolist(),c.tolist()))}")
