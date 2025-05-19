"""
t6B Classify EEG – version 0.1

Binary classifier (labels 0 vs 1) on windowed EEG only.
 • Loads a query-stage Zarr store (processed/queries/<query>.zarr)
 • Filters out windows whose label is not 0 or 1
 • Trains a small Conv1-D network
 • Reports train/val/test accuracy
 • Saves the model (.pth) and a metrics.json to S3
"""

import os
import sys
import json
import time
import math
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Base pipeline imports (relative for “script” mode, absolute for package)
try:
    from .base_transform import BaseTransform, Session   # when installed
    from .utils import create_data_loaders
except ImportError:
    from base_transform import BaseTransform, Session     # when run as script
    from utils import create_data_loaders


# --------------------------------------------------------------------- #
#  Hyper-parameter block – every tunable in one small dict
# --------------------------------------------------------------------- #
DEFAULT_HPARAMS = dict(
    epochs       = 20,
    batch_size   = 64,
    lr           = 3e-4,
    hidden       = 128,      # conv channels
    train_ratio  = 0.7,
    val_ratio    = 0.15,
    test_ratio   = 0.15,
    device       = "cuda" if torch.cuda.is_available() else "cpu",
    weight_decay = 1e-4,
)



# --------------------------------------------------------------------- #
#  Simple Conv1-D binary classifier
# --------------------------------------------------------------------- #
class EEGClassifier(nn.Module):
    def __init__(self,
                 in_channels: int,
                 seq_len: int,
                 hidden: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc   = nn.Linear(hidden, 2)

    def forward(self, x):                # x : (B,C,L)
        x = self.conv(x)
        x = self.pool(x).squeeze(-1)     # (B, hidden)
        return self.fc(x)                # logits



# --------------------------------------------------------------------- #
#  Transform
# --------------------------------------------------------------------- #
class ClassifyEEGTransform(BaseTransform):
    """Train a binary EEG classifier from a query Zarr store."""

    SOURCE_PREFIX = "processed/queries/"
    DEST_PREFIX   = "models/"

    # -------------------- life-cycle helpers --------------------------
    def __init__(self,
                 query_name: str,
                 hparams: Dict[str, Any],
                 **kwargs):
        self.query_name = query_name
        self.hp         = hparams

        super().__init__(**kwargs)

        self.logger.info(f"Query store : {self.query_name}.zarr")
        self.logger.info(f"Hyper-params: {self.hp}")

    # -------------------- CLI wiring ---------------------------------
    @classmethod
    def add_subclass_arguments(cls, parser: argparse.ArgumentParser):
        parser.add_argument("--query", dest="query_name",
                            default="eye_neural",
                            help="Name of query Zarr (without .zarr)")
        # one flag per h-param
        parser.add_argument("--epochs",      type=int,   default=DEFAULT_HPARAMS["epochs"])
        parser.add_argument("--lr",          type=float, default=DEFAULT_HPARAMS["lr"])
        parser.add_argument("--hidden",      type=int,   default=DEFAULT_HPARAMS["hidden"])
        parser.add_argument("--device",      type=str,   default=DEFAULT_HPARAMS["device"],
                            choices=["cpu", "cuda"])
        parser.add_argument("--train-ratio", type=float, default=DEFAULT_HPARAMS["train_ratio"])
        parser.add_argument("--val-ratio",   type=float, default=DEFAULT_HPARAMS["val_ratio"])
        parser.add_argument("--test-ratio",  type=float, default=DEFAULT_HPARAMS["test_ratio"])

    @classmethod
    def from_args(cls, args):
        # collect h-params
        hp = dict(DEFAULT_HPARAMS)
        # copy scalar h-params **except batch_size** (handled separately)
        for k in hp.keys():
            if k == "batch_size":
                continue
            attr = k.replace("-", "_")
            if hasattr(args, attr):
                hp[k] = getattr(args, attr)
                
        # mini-batch comes from the pipeline-level --batch-size (if > 0)
        if getattr(args, "batch_size", 0) > 0:
            hp["batch_size"] = args.batch_size
            
        return cls(
            query_name=args.query_name,
            hparams=hp,
            transform_id="t6B_classify_eeg_v0",
            script_id="6B",
            script_name="classify_eeg",
            script_version="v0",
            source_prefix=getattr(args, "source_prefix", cls.SOURCE_PREFIX),
            destination_prefix=getattr(args, "dest_prefix", cls.DEST_PREFIX),
            s3_bucket=args.s3_bucket,
            verbose=args.verbose,
            log_file=args.log_file,
            dry_run=args.dry_run,
            keep_local=args.keep_local,
        )

    # -------------------- override BaseTransform hooks ---------------
    def find_sessions(self):
        """Return a pseudo-session list with just the query name."""
        return [self.query_name]

    # -------------------- core work ----------------------------------
    def process_session(self, session: Session) -> Dict[str, Any]:
        """Train and evaluate the classifier; upload artefacts."""
        # session_id may already include ".zarr"
        base_name   = session.session_id[:-5] if session.session_id.endswith(".zarr") else session.session_id
        query_zarr  = f"s3://{self.s3_bucket}/{self.source_prefix}{base_name}.zarr"
        local_model = Path(session.temp_dir) / "eeg_classifier.pth"
        local_json  = Path(session.temp_dir) / "metrics.json"
        s3_model_key= f"{self.destination_prefix}{session.session_id}_eeg_classifier.pth"
        s3_json_key = f"{self.destination_prefix}{session.session_id}_metrics.json"

        # ---------- 1. build DataLoaders -----------------------------
        self.logger.info("Building DataLoaders")
        tr_loader, va_loader, te_loader = create_data_loaders(
            zarr_path=query_zarr,
            modalities=["eeg"],
            batch_size=self.hp["batch_size"],
            train_ratio=self.hp["train_ratio"],
            val_ratio=self.hp["val_ratio"],
            test_ratio=self.hp["test_ratio"],
            shuffle=True,
            num_workers=4,
        )

        # filter masks (only 0 / 1)
        def mask_loader(loader: DataLoader):
            for x, y in loader:
                m = (y <= 1)
                if m.sum() == 0:
                    continue
                yield x[m], y[m]

        tr_loader = list(mask_loader(tr_loader))
        va_loader = list(mask_loader(va_loader))
        te_loader = list(mask_loader(te_loader))

        if not tr_loader:
            return dict(status="skipped", metadata=dict(reason="no_binary_labels"))

        # infer shape
        sample, _ = tr_loader[0]
        _, C, L = sample.shape

        # ---------- 2. build model / optimiser -----------------------
        device = torch.device(self.hp["device"])
        model  = EEGClassifier(C, L, hidden=self.hp["hidden"]).to(device)
        opt    = torch.optim.AdamW(model.parameters(),
                                   lr=self.hp["lr"],
                                   weight_decay=self.hp["weight_decay"])
        loss_fn= nn.CrossEntropyLoss()

        # ---------- 3. training loop ---------------------------------
        best_acc = 0.0
        history  = []

        total_batches = len(tr_loader)
        for epoch in range(1, self.hp["epochs"] + 1):
            model.train()
            running_loss, seen = 0.0, 0
            for xb, yb in tr_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits  = model(xb)
                loss    = loss_fn(logits, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
                running_loss += loss.item() * yb.size(0)
                seen         += yb.size(0)

            train_loss = running_loss / seen
            val_acc    = self._eval_accuracy(model, va_loader, device)

            history.append(dict(epoch=epoch,
                                train_loss=train_loss,
                                val_acc=val_acc))
            self.logger.info(f"Epoch {epoch:02}/{self.hp['epochs']:02}  "
                             f"train_loss={train_loss:.4f}  val_acc={val_acc:.3f}")

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), local_model)

        # ---------- 4. final test ------------------------------------
        model.load_state_dict(torch.load(local_model))
        test_acc = self._eval_accuracy(model, te_loader, device)
        self.logger.info(f"BEST val_acc={best_acc:.3f}  TEST acc={test_acc:.3f}")

        # ---------- 5. save metrics ----------------------------------
        metrics = dict(
            best_val_accuracy = best_acc,
            test_accuracy     = test_acc,
            history           = history,
            hyperparams       = self.hp,
            trained_at        = time.strftime("%Y-%m-%d %H:%M:%S"),
        )
        with open(local_json, "w") as f:
            json.dump(metrics, f, indent=2)

        # ---------- 6. return upload instructions --------------------
        return dict(
            status="success",
            files_to_upload=[
                (str(local_model), s3_model_key),
                (str(local_json),  s3_json_key),
            ],
            files_to_copy=[],
            zarr_stores=[],
            metadata=dict(
                best_val_acc=best_acc,
                test_acc=test_acc,
                epochs=self.hp["epochs"],
                batch_size=self.hp["batch_size"],
            )
        )

    # -------------------- helper ------------------------------------
    @staticmethod
    def _eval_accuracy(model: nn.Module,
                       loader: List[Tuple[torch.Tensor, torch.Tensor]],
                       device: torch.device) -> float:
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb).argmax(dim=1)
                correct += (pred == yb).sum().item()
                total   += yb.size(0)
        return correct / total if total else math.nan



# --------------------------------------------------------------------- #
#  CLI entry-point
# --------------------------------------------------------------------- #
if __name__ == "__main__":
    ClassifyEEGTransform.run_from_command_line()
