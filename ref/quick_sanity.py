#!/usr/bin/env python
"""
quick_inspect.py
================
Reads a query-stage Zarr store, builds DataLoaders via utils.create_data_loaders,
and prints an ASCII-only summary that is easy to read in Windows cmd.

Information reported
--------------------
* Zarr path, number of sessions, total windows
* Which modalities are present per session
* Unique window shapes for EEG and fNIRS
* Label map stored in root.attrs
* Label distribution (counts and percent) across ALL windows
* Train / validation / test split sizes, batch size, number of batches
* A single sample batch shape check

All warnings about the unofficial vlen-bytes codec are harmless and ignored.
"""

import argparse
import sys
import warnings
from collections import Counter, defaultdict
from typing import List

import numpy as np
import torch
import zarr

# Silence the "vlen-bytes codec not in v3 spec" warning lines
warnings.filterwarnings("ignore", category=UserWarning, module="zarr.codecs.vlen_utf8")

# Import the DataLoader factory re-exported by utils/__init__.py
from utils import create_data_loaders


def gather_store_stats(root: zarr.Group, requested_modalities: List[str]):
    """
    Walk every session in the query store and collect:
      * window count
      * label array
      * window shapes
      * which modalities exist
    Returns a dict with aggregated information.
    """
    sessions_group = root["sessions"]
    stats = {
        "session_count": 0,
        "total_windows": 0,
        "labels": [],
        "eeg_shapes": set(),
        "fnirs_shapes": set(),
        "modalities_present": defaultdict(set),  # {modality: set(session_ids)}
    }

    for sid in sessions_group:
        sg = sessions_group[sid]
        stats["session_count"] += 1

        # Labels
        if "labels" not in sg:
            continue
        lbls = sg["labels"][:]
        if lbls.dtype.kind in ("S", "a"):
            lbls = lbls.astype(str)
        stats["labels"].append(lbls)
        n_windows = lbls.shape[0]
        stats["total_windows"] += n_windows

        # Modalities
        if "eeg" in requested_modalities and "eeg" in sg:
            stats["modalities_present"]["eeg"].add(sid)
            stats["eeg_shapes"].add(tuple(sg["eeg"].shape[1:]))  # (channels, samples)
        if "fnirs" in requested_modalities and "fnirs" in sg:
            stats["modalities_present"]["fnirs"].add(sid)
            shp = sg["fnirs"].shape[1:]
            # If saved as (channels,) store length only
            stats["fnirs_shapes"].add(tuple(shp) if len(shp) else ("channels_only",))

    # Flatten label arrays once at the end
    if stats["labels"]:
        stats["labels"] = np.concatenate(stats["labels"])
    else:
        stats["labels"] = np.array([], dtype=str)

    return stats


def print_header(title: str):
    line = "-" * len(title)
    print("\n" + title)
    print(line)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--zarr", required=True, help="Query Zarr store (S3 URI or path)")
    parser.add_argument("--modalities", nargs="+", default=["eeg", "fnirs"],
                        choices=["eeg", "fnirs"], help="Modalities to load")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers")
    args = parser.parse_args(argv)

    # ------------------------------------------------------------------ #
    #   Open Zarr store and gather statistics
    # ------------------------------------------------------------------ #
    print_header(">>> OPENING ZARR STORE")
    print("Store path       :", args.zarr)
    try:
        root = zarr.open_group(store=args.zarr, mode="r", storage_options={"anon": False})
    except Exception as e:
        sys.exit("ERROR: Could not open Zarr store. " + str(e))

    if "sessions" not in root:
        sys.exit("ERROR: 'sessions' group not found in store; is this a query output?")

    # Label map (if any)
    label_map = root.attrs.get("label_map", {})
    if not label_map:
        label_map = {"closed": 0, "open": 1, "intro": 2, "unknown": 3}

    # Gather per-session stats
    stats = gather_store_stats(root, args.modalities)

    print("Sessions found   :", stats["session_count"])
    print("Total windows    :", stats["total_windows"])

    # Modalities present
    for mod in args.modalities:
        present = "yes" if stats["modalities_present"][mod] else "no"
        print(f"{mod.upper()} present  :", present)

    # Unique shapes
    if stats["eeg_shapes"]:
        print("EEG window shape :", ", ".join(str(s) for s in sorted(stats["eeg_shapes"])))
    if stats["fnirs_shapes"]:
        print("fNIRS shape      :", ", ".join(str(s) for s in sorted(stats["fnirs_shapes"])))

    # ------------------------------------------------------------------ #
    #   Label distribution
    # ------------------------------------------------------------------ #
    print_header(">>> LABEL DISTRIBUTION")
    if stats["labels"].size == 0:
        print("No labels found.")
    else:
        # Convert possible bytes to str first (already handled)
        counter = Counter(stats["labels"].tolist())
        for label, count in counter.most_common():
            pct = 100.0 * count / stats["total_windows"] if stats["total_windows"] else 0.0
            print(f"{label:>10} : {count:6d}  ({pct:5.1f} %)")

    # ------------------------------------------------------------------ #
    #   Build DataLoaders and sample a batch
    # ------------------------------------------------------------------ #
    print_header(">>> BUILDING DATALOADERS")
    try:
        train_loader, val_loader, test_loader = create_data_loaders(
            zarr_path=args.zarr,
            batch_size=args.batch,
            modalities=args.modalities,
            num_workers=args.workers,
            shuffle=True,
        )
    except ValueError as err:
        sys.exit("ERROR while creating loaders: " + str(err))

    train_size = len(train_loader.dataset)
    val_size   = len(val_loader.dataset) if val_loader else 0
    test_size  = len(test_loader.dataset) if test_loader else 0
    print("Train windows    :", train_size)
    print("Val windows      :", val_size)
    print("Test windows     :", test_size)
    print("Batch size       :", args.batch)
    print("Train batches    :", len(train_loader))

    # Pull a single batch
    print_header(">>> FIRST BATCH SHAPES")
    batch_feat, batch_lab = next(iter(train_loader))
    bsz = batch_lab.size(0)
    print("Labels tensor    :", tuple(batch_lab.shape), str(batch_lab.dtype))

    if len(args.modalities) == 1:
        tensor = batch_feat
        print(f"{args.modalities[0].upper()} tensor :", tuple(tensor.shape), str(tensor.dtype))
    else:
        for mod in args.modalities:
            tensor = batch_feat[mod]
            print(f"{mod.upper()} tensor :", tuple(tensor.shape), str(tensor.dtype))

    print_header(">>> SUMMARY COMPLETE")


if __name__ == "__main__":
    main()
