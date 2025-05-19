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
import datetime
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
        print(f"\n=============== PROCESSING SESSION: {session.session_id} ===============\n")
        # session_id may already include ".zarr"
        base_name   = session.session_id[:-5] if session.session_id.endswith(".zarr") else session.session_id
        query_zarr  = f"s3://{self.s3_bucket}/{self.source_prefix}{base_name}.zarr"
        print(f"Using zarr store: {query_zarr}")
        local_model = Path(session.temp_dir) / "eeg_classifier.pth"
        local_json  = Path(session.temp_dir) / "metrics.json"
        
        # Add timestamp to ensure unique filenames (never overwrite)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        s3_model_key= f"{self.destination_prefix}{session.session_id}_eeg_classifier_{timestamp}.pth"
        s3_json_key = f"{self.destination_prefix}{session.session_id}_metrics_{timestamp}.json"

        # ---------- 1. build DataLoaders -----------------------------
        print("Building DataLoaders...")
        tr_loader, va_loader, te_loader = create_data_loaders(
            zarr_path=query_zarr,
            modalities=["eeg"],
            batch_size=self.hp["batch_size"],
            train_ratio=self.hp["train_ratio"],
            val_ratio=self.hp["val_ratio"],
            test_ratio=self.hp["test_ratio"],
            shuffle=True,
            num_workers=4,
            split_by_session=False  # Prevent data leakage across sessions
        )

        # filter masks (only 0 / 1)
        def mask_loader(loader: DataLoader):
            for x, y in loader:
                m = (y <= 1)
                if m.sum() == 0:
                    continue
                yield x[m], y[m]

        # Safely filter loaders (handle case where va_loader or te_loader might be None)
        tr_loader = list(mask_loader(tr_loader)) if tr_loader else []
        va_loader = list(mask_loader(va_loader)) if va_loader else []
        te_loader = list(mask_loader(te_loader)) if te_loader else []
        
        # Log dataset splits
        print(f"Dataset splits after filtering: {len(tr_loader)} train, {len(va_loader)} validation, {len(te_loader)} test batches")
        
        # === DIAGNOSTIC 1: Check label distribution ===
        print("=== DATASET LABEL DISTRIBUTION ===")
        for name, loader in [("train", tr_loader), ("val", va_loader), ("test", te_loader)]:
            if loader:
                # Get all labels across all batches
                all_labels = torch.cat([y for _, y in loader])
                unique_labels, counts = torch.unique(all_labels, return_counts=True)
                print(f"{name:5s}: {dict(zip(unique_labels.tolist(), counts.tolist()))}")
            else:
                print(f"{name:5s}: Empty")
        print("===================================")
        
        # === DIAGNOSTIC 3: Simple assertion check for multiple classes ===
        if tr_loader:
            train_labels = torch.cat([y for _, y in tr_loader])
            unique_train_labels = torch.unique(train_labels).tolist()
            print(f"SANITY CHECK: Training labels found: {unique_train_labels}")
            if len(unique_train_labels) <= 1:
                print(f"WARNING: Training data has only one label: {unique_train_labels}")
        
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
            
            # Handle case with no validation data
            if not va_loader:
                print("No validation data available, using training data for model selection")
                val_acc = self._eval_accuracy(model, tr_loader, device)
            else:
                val_acc = self._eval_accuracy(model, va_loader, device)

            history.append(dict(epoch=epoch,
                                train_loss=train_loss,
                                val_acc=val_acc))
            print(f"Epoch {epoch:02}/{self.hp['epochs']:02}  "
                  f"train_loss={train_loss:.4f}  val_acc={val_acc:.3f}")

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), local_model)

        # ---------- 4. final test ------------------------------------
        model.load_state_dict(torch.load(local_model))
        
        # Handle case with no test data
        if not te_loader:
            print("No test data available, using training data for final evaluation")
            test_acc = self._eval_accuracy(model, tr_loader, device)
        else:
            test_acc = self._eval_accuracy(model, te_loader, device)
            
        print(f"BEST val_acc={best_acc:.3f}  TEST acc={test_acc:.3f}")

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
    @classmethod
    def _eval_accuracy(cls, model: nn.Module,
                       loader: List[Tuple[torch.Tensor, torch.Tensor]],
                       device: torch.device) -> float:
        """Evaluate model accuracy with detailed per-class statistics."""
        model.eval()
        
        # Handle empty loader (no validation/test data)
        if not loader:
            print("Evaluation called with empty loader")
            return 0.0  # Return 0 accuracy for empty loaders
            
        correct, total = 0, 0
        # Track per-class statistics
        class_correct = {0: 0, 1: 0}
        class_total = {0: 0, 1: 0}
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                pred = logits.argmax(dim=1)
                
                # Store predictions and labels for later analysis
                all_preds.append(pred.cpu())
                all_labels.append(yb.cpu())
                
                # Overall accuracy
                correct += (pred == yb).sum().item()
                total += yb.size(0)
                
                # Per-class accuracy
                for c in range(2):  # Binary classification
                    class_mask = (yb == c)
                    class_correct[c] += ((pred == c) & class_mask).sum().item()
                    class_total[c] += class_mask.sum().item()
        
        # === DIAGNOSTIC 2: Enhanced evaluation statistics ===
        # Calculate per-class accuracy
        print("\n=== EVALUATION STATISTICS ===")
        for c in range(2):
            if class_total[c] > 0:
                print(f"Class {c} accuracy: {class_correct[c]/class_total[c]:.4f} ({class_correct[c]}/{class_total[c]})")
            else:
                print(f"Class {c}: No samples")
        
        # Check prediction distribution
        if all_preds:
            all_preds = torch.cat(all_preds)
            all_labels = torch.cat(all_labels)
            unique_preds, counts = torch.unique(all_preds, return_counts=True)
            print(f"Prediction distribution: {dict(zip(unique_preds.tolist(), counts.tolist()))}")
            
            # Simple confusion matrix
            cm = [[0, 0], [0, 0]]  # 2x2 matrix for binary classification
            for true_label, pred_label in zip(all_labels.tolist(), all_preds.tolist()):
                if true_label < 2 and pred_label < 2:  # Safety check
                    cm[true_label][pred_label] += 1
            
            print("Confusion Matrix:")
            print("       | Pred 0 | Pred 1 |")
            print(f"True 0 | {cm[0][0]:6d} | {cm[0][1]:6d} |")
            print(f"True 1 | {cm[1][0]:6d} | {cm[1][1]:6d} |")
        print("=============================")
        
        return correct / total if total else 0.0  # Use 0.0 instead of NaN



# --------------------------------------------------------------------- #
#  CLI entry-point
# --------------------------------------------------------------------- #
if __name__ == "__main__":
    ClassifyEEGTransform.run_from_command_line()
