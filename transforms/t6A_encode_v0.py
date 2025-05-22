"""
t6A Encode EEG – version 0.1

PLACEHOLDER transform for neural encoding, currently implements efficient binary classification.
This transform is designed to demonstrate efficient data processing and model training:
 • Loads a query-stage Zarr store (processed/queries/<query>.zarr)
 • Efficiently pre-processes labels and filters data upfront
 • Uses vectorized operations and caching for better performance
 • Trains an optimized EEG classifier model
 • Reports train/val/test accuracy with detailed metrics
 • Saves the model (.pth) and a metrics.json to S3

This transform will be updated to implement actual encoding models in the future.
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
from typing import List, Dict, Any, Optional, Tuple, Union, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Base pipeline imports (relative for "script" mode, absolute for package)
try:
    from .base_transform import BaseTransform, Session   # when installed
    from .transforms.encoding import (  # when installed
        create_efficient_data_loaders,
        filter_binary_labels,
        EfficientEEGClassifier,
    )
except ImportError:
    from base_transform import BaseTransform, Session     # when run as script
    from transforms.encoding import (  # when run as script
        create_efficient_data_loaders,
        filter_binary_labels,
        EfficientEEGClassifier,
    )


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
    dropout      = 0.2,
    num_layers   = 2,
    patience     = 5,        # early stopping patience
    use_mps      = torch.backends.mps.is_available(),  # Apple Silicon GPU
)


# --------------------------------------------------------------------- #
#  Transform
# --------------------------------------------------------------------- #
class EncodeEEGTransform(BaseTransform):
    """
    Train an efficient EEG encoder (currently classifier) from a query Zarr store.
    This is a placeholder transform that will be updated with actual encoding in future.
    """

    SOURCE_PREFIX = "processed/queries/"
    DEST_PREFIX   = "models/"

    # -------------------- life-cycle helpers --------------------------
    def __init__(self,
                 query_name: str,
                 num_classes: int = 2,
                 hparams: Dict[str, Any] = None,
                 verbose: bool = False,
                 **kwargs):
        self.query_name = query_name
        self.num_classes = num_classes
        self.verbose = verbose
        self.hp = hparams or dict(DEFAULT_HPARAMS)

        # Ensure device is correctly set
        if self.hp["use_mps"] and torch.backends.mps.is_available():
            self.hp["device"] = "mps"  # Apple Silicon GPU
        elif torch.cuda.is_available():
            self.hp["device"] = "cuda"
        else:
            self.hp["device"] = "cpu"

        super().__init__(**kwargs)

        self.logger.info(f"Query store: {self.query_name}.zarr")
        self.logger.info(f"Device: {self.hp['device']}")
        self.logger.info(f"Number of classes: {self.num_classes}")
        self.logger.info(f"Hyper-parameters: {self.hp}")

    # -------------------- CLI wiring ---------------------------------
    @classmethod
    def add_subclass_arguments(cls, parser: argparse.ArgumentParser):
        parser.add_argument("--query", dest="query_name",
                            default="eye_neural",
                            help="Name of query Zarr (without .zarr)")
        parser.add_argument("--num-classes", type=int, default=2,
                            help="Number of class labels to use")
        # Removed duplicate --verbose argument that's already defined at the main parser level
                            
        # Add hyperparameter arguments
        parser.add_argument("--epochs", type=int, default=DEFAULT_HPARAMS["epochs"])
        parser.add_argument("--lr", type=float, default=DEFAULT_HPARAMS["lr"])
        parser.add_argument("--hidden", type=int, default=DEFAULT_HPARAMS["hidden"])
        parser.add_argument("--dropout", type=float, default=DEFAULT_HPARAMS["dropout"])
        parser.add_argument("--num-layers", type=int, default=DEFAULT_HPARAMS["num_layers"])
        parser.add_argument("--patience", type=int, default=DEFAULT_HPARAMS["patience"],
                           help="Early stopping patience (0 to disable)")
        parser.add_argument("--train-ratio", type=float, default=DEFAULT_HPARAMS["train_ratio"])
        parser.add_argument("--val-ratio", type=float, default=DEFAULT_HPARAMS["val_ratio"])
        parser.add_argument("--test-ratio", type=float, default=DEFAULT_HPARAMS["test_ratio"])
        parser.add_argument("--no-mps", action="store_true",
                           help="Disable MPS (Apple Silicon GPU) even if available")

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
                
        # Special handling for MPS flag
        if hasattr(args, "no_mps") and args.no_mps:
            hp["use_mps"] = False
                
        # mini-batch comes from the pipeline-level --batch-size (if > 0)
        if getattr(args, "batch_size", 0) > 0:
            hp["batch_size"] = args.batch_size
            
        return cls(
            query_name=args.query_name,
            num_classes=args.num_classes,
            hparams=hp,
            verbose=args.verbose if hasattr(args, "verbose") else False,
            transform_id="t6A_encode_v0",
            script_id="6A",
            script_name="encode_eeg",
            script_version="v0",
            source_prefix=getattr(args, "source_prefix", cls.SOURCE_PREFIX),
            destination_prefix=getattr(args, "dest_prefix", cls.DEST_PREFIX),
            s3_bucket=args.s3_bucket,
            log_file=args.log_file,
            dry_run=args.dry_run,
            keep_local=args.keep_local,
        )

    # -------------------- override BaseTransform hooks ---------------
    def find_sessions(self):
        """Return a list with just the query name."""
        return [self.query_name]

    # -------------------- core work ----------------------------------
    def process_session(self, session: Session) -> Dict[str, Any]:
        """Train and evaluate the model; upload artifacts."""
        self.logger.info(f"\n=============== PROCESSING SESSION: {session.session_id} ===============\n")
        
        # session_id may already include ".zarr"
        base_name   = session.session_id[:-5] if session.session_id.endswith(".zarr") else session.session_id
        query_zarr  = f"s3://{self.s3_bucket}/{self.source_prefix}{base_name}.zarr"
        self.logger.info(f"Using zarr store: {query_zarr}")
        
        # Create output files with timestamp for uniqueness
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        local_model = Path(session.temp_dir) / "eeg_encoder.pth"
        local_json  = Path(session.temp_dir) / "metrics.json"
        s3_model_key= f"{self.destination_prefix}{session.session_id}_eeg_encoder_{timestamp}.pth"
        s3_json_key = f"{self.destination_prefix}{session.session_id}_metrics_{timestamp}.json"

        # ---------- 1. build DataLoaders -----------------------------
        self.logger.info("Building optimized DataLoaders...")
        
        # Define filter function based on num_classes
        # For binary classification, we want only labels 0 and 1
        # For multiclass, we want labels 0 through num_classes-1
        filter_func = None
        if self.num_classes == 2:
            filter_func = filter_binary_labels
        elif self.num_classes > 2:
            filter_func = filter_labels_in_range(0, self.num_classes - 1)
        
        try:
            tr_loader, va_loader, te_loader = create_efficient_data_loaders(
                zarr_path=query_zarr,
                modalities=["eeg"],  # Currently only using EEG
                batch_size=self.hp["batch_size"],
                train_ratio=self.hp["train_ratio"],
                val_ratio=self.hp["val_ratio"],
                test_ratio=self.hp["test_ratio"],
                shuffle=True,
                num_workers=4,
                filter_label_func=filter_func,
                preload_labels=True,  # Major optimization
                split_by_session=True,  # Prevent data leakage
                verbose=self.verbose
            )
        except Exception as e:
            self.logger.error(f"Error creating data loaders: {e}")
            return {
                "status": "failed",
                "error_details": f"Data loader creation failed: {str(e)}",
                "metadata": {"error": str(e)}
            }

        # Check if we have data
        if not tr_loader or len(tr_loader) == 0:
            self.logger.warning("No training data available after filtering")
            return {
                "status": "skipped",
                "metadata": {"reason": "no_valid_training_data"}
            }

        # Get a sample batch to determine input shape
        inputs, _ = next(iter(tr_loader))
        if isinstance(inputs, dict):
            # Multi-modal data
            input_shape = {k: v.shape for k, v in inputs.items()}
            eeg_shape = input_shape["eeg"][1:]  # Remove batch dimension
            self.logger.info(f"Input shape: {input_shape}")
            in_channels, seq_len = eeg_shape
        else:
            # Single modality (EEG)
            in_channels, seq_len = inputs.shape[1:]
            self.logger.info(f"Input shape: (B, {in_channels}, {seq_len})")

        # ---------- 2. build model / optimiser -----------------------
        device = torch.device(self.hp["device"])
        
        model = EfficientEEGClassifier(
            in_channels=in_channels,
            seq_len=seq_len,
            hidden_size=self.hp["hidden"],
            num_layers=self.hp["num_layers"],
            dropout=self.hp["dropout"],
            num_classes=self.num_classes
        ).to(device)
        
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.hp["lr"],
            weight_decay=self.hp["weight_decay"]
        )
        
        # Use class weights if we have binary classification
        if self.num_classes == 2:
            # Count class distribution in training data
            class_counts = torch.zeros(self.num_classes)
            for _, labels in tr_loader:
                for i in range(self.num_classes):
                    class_counts[i] += (labels == i).sum().item()
                    
            if (class_counts > 0).all():
                # Compute inverse class weights
                class_weights = 1.0 / class_counts
                class_weights = class_weights / class_weights.sum()
                class_weights = class_weights.to(device)
                self.logger.info(f"Using class weights: {class_weights}")
                loss_fn = nn.CrossEntropyLoss(weight=class_weights)
            else:
                loss_fn = nn.CrossEntropyLoss()
        else:
            loss_fn = nn.CrossEntropyLoss()

        # ---------- 3. training loop ---------------------------------
        best_acc = 0.0
        best_epoch = 0
        history = []
        patience_counter = 0

        total_batches = len(tr_loader)
        self.logger.info(f"Starting training with {total_batches} batches per epoch")
        
        for epoch in range(1, self.hp["epochs"] + 1):
            # Training phase
            model.train()
            running_loss = 0.0
            running_acc = 0.0
            num_samples = 0
            
            for inputs, labels in tr_loader:
                # Move data to device
                if isinstance(inputs, dict):
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                else:
                    inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Track metrics
                running_loss += loss.item() * labels.size(0)
                preds = outputs.argmax(dim=1)
                running_acc += (preds == labels).sum().item()
                num_samples += labels.size(0)
            
            train_loss = running_loss / num_samples
            train_acc = running_acc / num_samples
            
            # Validation phase
            val_acc, val_metrics = self._evaluate_model(model, va_loader, device)
            
            # Log progress
            self.logger.info(f"Epoch {epoch:02d}/{self.hp['epochs']:02d} - "
                           f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                           f"Val Acc: {val_acc:.4f}")
            
            # Record history
            history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "val_metrics": val_metrics
            })
            
            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                best_epoch = epoch
                patience_counter = 0
                torch.save(model.state_dict(), local_model)
                self.logger.info(f"New best model saved (acc: {best_acc:.4f})")
            else:
                patience_counter += 1
                
            # Early stopping
            if self.hp["patience"] > 0 and patience_counter >= self.hp["patience"]:
                self.logger.info(f"Early stopping at epoch {epoch} (no improvement for {patience_counter} epochs)")
                break

        # ---------- 4. final test ------------------------------------
        self.logger.info("Loading best model for final evaluation")
        model.load_state_dict(torch.load(local_model))
        model.eval()
        
        # Test on the test set
        test_acc, test_metrics = self._evaluate_model(model, te_loader, device)
        self.logger.info(f"Best model from epoch {best_epoch}")
        self.logger.info(f"Best validation accuracy: {best_acc:.4f}")
        self.logger.info(f"Test accuracy: {test_acc:.4f}")
        
        # ---------- 5. save metrics ----------------------------------
        metrics = {
            "best_validation_accuracy": best_acc,
            "best_epoch": best_epoch,
            "test_accuracy": test_acc,
            "test_metrics": test_metrics,
            "training_history": history,
            "hyperparameters": self.hp,
            "model_info": {
                "type": "EfficientEEGClassifier",
                "in_channels": in_channels,
                "seq_len": seq_len,
                "hidden_size": self.hp["hidden"],
                "num_layers": self.hp["num_layers"],
                "dropout": self.hp["dropout"],
                "num_classes": self.num_classes
            },
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # Save metrics to JSON
        with open(local_json, "w") as f:
            json.dump(metrics, f, indent=2)

        # ---------- 6. return upload instructions --------------------
        return {
            "status": "success",
            "files_to_upload": [
                (str(local_model), s3_model_key),
                (str(local_json), s3_json_key),
            ],
            "files_to_copy": [],
            "zarr_stores": [],
            "metadata": {
                "best_val_acc": best_acc,
                "test_acc": test_acc,
                "epochs_trained": epoch,
                "best_epoch": best_epoch,
                "batch_size": self.hp["batch_size"],
                "model_type": "EfficientEEGClassifier",
                "num_classes": self.num_classes
            }
        }

    # -------------------- evaluation helpers ------------------------
    def _evaluate_model(self, model: nn.Module, 
                        loader: DataLoader, 
                        device: torch.device) -> Tuple[float, Dict]:
        """
        Evaluate model with detailed metrics.
        
        Args:
            model: The PyTorch model to evaluate
            loader: DataLoader with evaluation data
            device: Device to run evaluation on
            
        Returns:
            Tuple of (accuracy, metrics_dict)
        """
        if not loader:
            self.logger.warning("Evaluation called with empty loader")
            return 0.0, {}
            
        model.eval()
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in loader:
                # Move data to device
                if isinstance(inputs, dict):
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                else:
                    inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Forward pass
                outputs = model(inputs)
                preds = outputs.argmax(dim=1)
                
                # Store predictions and labels for analysis
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())
        
        # Concatenate all batches
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        
        # Calculate accuracy
        correct = (all_preds == all_labels).sum().item()
        total = all_labels.size(0)
        accuracy = correct / total if total > 0 else 0.0
        
        # Calculate per-class metrics
        metrics = {}
        
        # Per-class accuracy
        for class_idx in range(self.num_classes):
            class_mask = (all_labels == class_idx)
            class_total = class_mask.sum().item()
            
            if class_total > 0:
                class_correct = ((all_preds == class_idx) & class_mask).sum().item()
                metrics[f"class_{class_idx}_accuracy"] = class_correct / class_total
                metrics[f"class_{class_idx}_count"] = class_total
            else:
                metrics[f"class_{class_idx}_accuracy"] = 0.0
                metrics[f"class_{class_idx}_count"] = 0
        
        # Confusion matrix (for detailed analysis)
        confusion_matrix = torch.zeros(self.num_classes, self.num_classes, dtype=torch.int)
        for t, p in zip(all_labels, all_preds):
            confusion_matrix[t, p] += 1
        
        metrics["confusion_matrix"] = confusion_matrix.tolist()
        metrics["accuracy"] = accuracy
        metrics["total_samples"] = total
        
        # Print detailed metrics if verbose
        if self.verbose:
            self.logger.info("\n=== EVALUATION METRICS ===")
            for class_idx in range(self.num_classes):
                self.logger.info(f"Class {class_idx} accuracy: {metrics[f'class_{class_idx}_accuracy']:.4f} "
                               f"({metrics[f'class_{class_idx}_count']} samples)")
            
            # Print confusion matrix
            self.logger.info("Confusion Matrix:")
            self.logger.info(confusion_matrix)
            self.logger.info("==========================")
            
        return accuracy, metrics


# --------------------------------------------------------------------- #
#  CLI entry-point
# --------------------------------------------------------------------- #
if __name__ == "__main__":
    EncodeEEGTransform.run_from_command_line()