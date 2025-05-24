"""
T6C Baseline EEG Classifier – version 0

Trains a baseline CNN classifier on EEG data from query zarr stores.
Designed to work with outputs from t4D query transforms (e.g., eye_good).

• Uses session-wise data splitting to prevent leakage
• Automatic class imbalance handling via weighted sampling
• Saves trained model weights and evaluation metrics
• Supports both binary and multi-class classification

Results are saved to:
    processed/models/baseline_<query_name>_<timestamp>.zarr

Root attributes include:
    ├─ model_type = "baseline_cnn"
    ├─ query_source = source zarr path
    ├─ train_accuracy
    ├─ val_accuracy  
    ├─ test_accuracy
    └─ class_distribution

Usage Examples:

Basic training on eye_good query results:
```bash
python transforms/t6C_baseline_v0.py \
    --query-name eye_good \
    --s3-bucket conduit-data-dev \
    --num-epochs 50 \
    --train-batch-size 32
```

Fast training for testing (skips analysis):
```bash
python transforms/t6C_baseline_v0.py \
    --query-name eye_good \
    --s3-bucket conduit-data-dev \
    --num-epochs 5 \
    --train-batch-size 32 \
    --skip-analysis \
    --no-auto-balance \
    --dry-run
```

Ultra-fast testing (limited sessions):
```bash
python transforms/t6C_baseline_v0.py \
    --query-name eye_good \
    --s3-bucket conduit-data-dev \
    --num-epochs 2 \
    --train-batch-size 16 \
    --skip-analysis \
    --no-auto-balance \
    --session-limit 3 \
    --dry-run
```

Training with custom hyperparameters:
```bash
python transforms/t6C_baseline_v0.py \
    --query-name eye_good \
    --learning-rate 0.0001 \
    --num-epochs 100 \
    --train-batch-size 64 \
    --normalize-features \
    --early-stopping-patience 15 \
    --train-ratio 0.8 \
    --val-ratio 0.1 \
    --test-ratio 0.1
```
</edits>

Programmatic usage:
```python
from transforms.t6C_baseline_v0 import BaselineClassifierTransform

# Create and run classifier
classifier = BaselineClassifierTransform(
    query_name="eye_good",
    learning_rate=0.001,
    num_epochs=50,
    normalize_features=True,
    s3_bucket="conduit-data-dev"
)

# Find and process query results
sessions = classifier.find_sessions()
for session_id in sessions:
    session = classifier.create_session(session_id)
    result = classifier.process_session(session)
    print(f"Training result: {result['metadata']['test_accuracy']:.3f}")
```

The transform automatically:
- Loads query data using the enhanced dataloader
- Splits data by session to prevent leakage
- Handles class imbalance with weighted sampling
- Trains a CNN with early stopping
- Evaluates on held-out test set
- Saves model weights and metrics to zarr

Output zarr structure:
- /model/* - Model state dict parameters
- /history/* - Training history (loss, accuracy per epoch)
- /normalization/* - Feature normalization stats (if enabled)
- Root attrs - Metadata, hyperparameters, final metrics
"""

# --------------------------------------------------------------------- #
# Std-lib & third-party
# --------------------------------------------------------------------- #
import json
import logging
import sys
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import zarr

# Simple implementations to avoid sklearn dependency
def accuracy_score(y_true, y_pred):
    """Calculate accuracy score with proper type handling."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    if len(y_true) == 0:
        return 0.0
    accuracy = np.mean(y_true == y_pred)
    return float(accuracy)

def classification_report(y_true, y_pred, output_dict=False):
    # Simple classification report implementation
    classes = np.unique(np.concatenate([y_true, y_pred]))
    report = {}
    
    for cls in classes:
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        report[str(cls)] = {'precision': precision, 'recall': recall, 'f1-score': f1}
    
    if output_dict:
        return report
    else:
        # Return string representation
        lines = []
        for cls, metrics in report.items():
            lines.append(f"Class {cls}: precision={metrics['precision']:.3f}, recall={metrics['recall']:.3f}, f1={metrics['f1-score']:.3f}")
        return '\n'.join(lines)

def confusion_matrix(y_true, y_pred):
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(classes)
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    
    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    
    for true, pred in zip(y_true, y_pred):
        matrix[class_to_idx[true], class_to_idx[pred]] += 1
    
    return matrix

# --------------------------------------------------------------------- #
# Pipeline-internal imports
# --------------------------------------------------------------------- #
from base_transform import BaseTransform, Session
from transforms.encoding.dataloader import create_neural_data_loaders

# --------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------- #
logger = logging.getLogger(__name__)


# ===================================================================== #
#  Model Definitions
# ===================================================================== #
class BaselineCNN(nn.Module):
    """
    Simple baseline CNN for EEG classification.
    
    Architecture:
    - 1D convolutions over time dimension
    - Batch normalization and dropout for regularization
    - Global average pooling to handle variable sequence lengths
    - Fully connected classification head
    """
    
    def __init__(self, n_channels: int, n_classes: int, sequence_length: int):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.sequence_length = sequence_length
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(n_channels, 32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classification head
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, n_classes)
        
    def forward(self, x):
        # Input shape: (batch, channels, time)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool1d(x, 2)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool1d(x, 2)
        
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Global average pooling
        x = self.global_pool(x)  # (batch, 128, 1)
        x = x.squeeze(-1)  # (batch, 128)
        
        # Classification
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


# ===================================================================== #
#  Transform class
# ===================================================================== #
class BaselineClassifierTransform(BaseTransform):
    """
    Trains a baseline CNN classifier on EEG data from query zarr stores.
    
    • Loads data using enhanced dataloader with session-wise splitting
    • Trains a simple CNN with standard hyperparameters
    • Evaluates on held-out test set
    • Saves model weights and metrics to zarr store
    """

    SOURCE_PREFIX = "processed/queries/"
    DEST_PREFIX   = "processed/models/"

    # ------------------------------------------------------------------ #
    # constructor / CLI helper
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        query_name: str = "eye_good",
        model_type: str = "baseline_cnn",
        batch_size: int = 32,
        learning_rate: float = 0.001,
        num_epochs: int = 50,
        early_stopping_patience: int = 10,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        normalize_features: bool = True,
        device: Optional[str] = None,
        seed: int = 42,
        skip_analysis: bool = False,
        auto_balance: bool = True,
        disable_preloading: bool = False,
        session_limit: Optional[int] = None,
        verbose_training: bool = False,
        **kwargs,
    ):
        self.query_name = query_name
        self.model_type = model_type
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.early_stopping_patience = early_stopping_patience
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.should_normalize = normalize_features
        self.seed = seed
        self.skip_analysis = skip_analysis
        self.auto_balance = auto_balance
        self.disable_preloading = disable_preloading
        self.session_limit = session_limit
        self.verbose_training = verbose_training
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        transform_id = kwargs.pop(
            "transform_id",
            f"t6C_baseline_v0_{query_name}_{model_type}",
        )
        script_id      = kwargs.pop("script_id", "6C")
        script_name    = kwargs.pop("script_name", "baseline_classifier")
        script_version = kwargs.pop("script_version", "v0")

        super().__init__(
            transform_id       = transform_id,
            script_id          = script_id,
            script_name        = script_name,
            script_version     = script_version,
            **kwargs,
        )

        self.logger.info(
            f"BaselineClassifierTransform initialized "
            f"(query_name={self.query_name}, "
            f"model_type={self.model_type}, "
            f"device={self.device})"
        )

    # ----------------------------- CLI hooks -------------------------- #
    @classmethod
    def add_subclass_arguments(cls, parser) -> None:
        parser.add_argument(
            "--query-name",
            type=str,
            default="eye_good",
            help="Name of query to train on (used to find zarr files) (default: eye_good)",
        )
        parser.add_argument(
            "--model-type",
            type=str,
            default="baseline_cnn",
            help="Type of model to train (default: baseline_cnn)",
        )
        parser.add_argument(
            "--train-batch-size",
            type=int,
            default=32,
            help="Batch size for training (default: 32)",
        )
        parser.add_argument(
            "--learning-rate",
            type=float,
            default=0.001,
            help="Learning rate (default: 0.001)",
        )
        parser.add_argument(
            "--num-epochs",
            type=int,
            default=50,
            help="Maximum number of training epochs (default: 50)",
        )
        parser.add_argument(
            "--early-stopping-patience",
            type=int,
            default=10,
            help="Early stopping patience (default: 10)",
        )
        parser.add_argument(
            "--train-ratio",
            type=float,
            default=0.7,
            help="Fraction of data for training (default: 0.7)",
        )
        parser.add_argument(
            "--val-ratio",
            type=float,
            default=0.15,
            help="Fraction of data for validation (default: 0.15)",
        )
        parser.add_argument(
            "--test-ratio",
            type=float,
            default=0.15,
            help="Fraction of data for testing (default: 0.15)",
        )
        parser.add_argument(
            "--normalize-features",
            action="store_true",
            help="Whether to normalize features (z-score)",
        )
        parser.add_argument(
            "--device",
            type=str,
            help="Device to use for training (cuda/cpu, default: auto)",
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=42,
            help="Random seed (default: 42)",
        )
        parser.add_argument(
            "--skip-analysis",
            action="store_true",
            help="Skip label distribution analysis for faster startup",
        )
        parser.add_argument(
            "--no-auto-balance",
            action="store_true",
            help="Disable automatic class balancing",
        )
        parser.add_argument(
            "--disable-preloading",
            action="store_true",
            help="Disable label preloading for ultra-fast testing (slower training)",
        )
        parser.add_argument(
            "--session-limit",
            type=int,
            help="Limit number of sessions processed for fast testing",
        )
        parser.add_argument(
            "--verbose-training",
            action="store_true",
            help="Enable verbose debug logging during training",
        )

    @classmethod
    def from_args(cls, args) -> "BaselineClassifierTransform":
        return cls(
            query_name                = getattr(args, "query_name", "eye_good"),
            model_type                = getattr(args, "model_type", "baseline_cnn"),
            batch_size                = getattr(args, "train_batch_size", 32),
            learning_rate             = getattr(args, "learning_rate", 0.001),
            num_epochs                = getattr(args, "num_epochs", 50),
            early_stopping_patience   = getattr(args, "early_stopping_patience", 10),
            train_ratio               = getattr(args, "train_ratio", 0.7),
            val_ratio                 = getattr(args, "val_ratio", 0.15),
            test_ratio                = getattr(args, "test_ratio", 0.15),
            normalize_features        = getattr(args, "normalize_features", False),
            device                    = getattr(args, "device", None),
            seed                      = getattr(args, "seed", 42),
            skip_analysis             = getattr(args, "skip_analysis", False),
            auto_balance              = not getattr(args, "no_auto_balance", False),
            disable_preloading        = getattr(args, "disable_preloading", False),
            session_limit             = getattr(args, "session_limit", None),
            verbose_training          = getattr(args, "verbose_training", False),
            source_prefix             = getattr(args, "source_prefix", cls.SOURCE_PREFIX),
            destination_prefix        = getattr(args, "dest_prefix",    cls.DEST_PREFIX),
            s3_bucket                 = args.s3_bucket,
            verbose                   = args.verbose,
            log_file                  = args.log_file,
            dry_run                   = args.dry_run,
            keep_local                = getattr(args, "keep_local", False),
        )

    # ------------------------------------------------------------------ #
    # session discovery
    # ------------------------------------------------------------------ #
    def find_sessions(self) -> List[str]:
        """
        Find query zarr stores that match the specified query name.
        """
        bkt = self.s3_bucket
        sessions = []
        
        # Look for zarr stores in the queries directory
        resp = self.s3.list_objects_v2(Bucket=bkt, Prefix=self.SOURCE_PREFIX)
        for o in resp.get("Contents", []):
            k = o["Key"]
            if k.endswith("zarr.json") and self.query_name in k:
                # Extract zarr store name (remove /zarr.json suffix)
                zarr_name = k.split("/")[-2].replace(".zarr", "")
                sessions.append(zarr_name)
        
        sessions = sorted(set(sessions))  # Remove duplicates
        self.logger.info(f"Found {len(sessions)} query zarr stores matching '{self.query_name}'")
        return sessions

    # ------------------------------------------------------------------ #
    # training utilities
    # ------------------------------------------------------------------ #
    def compute_normalization_stats(self, train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute normalization statistics from training data and return mean/std.
        """
        self.logger.info("Computing normalization statistics from training data")
        
        # Collect all training features
        all_features = []
        for batch_features, _ in train_loader:
            if isinstance(batch_features, dict):
                # Multi-modal case - use EEG
                features = batch_features.get("eeg", batch_features[list(batch_features.keys())[0]])
            else:
                features = batch_features
            all_features.append(features)
        
        # Concatenate and compute statistics
        all_features = torch.cat(all_features, dim=0)  # (N, channels, time)
        
        # Compute mean and std across samples and time, keeping channel dimension
        mean = all_features.mean(dim=(0, 2), keepdim=True)  # (1, channels, 1)
        std = all_features.std(dim=(0, 2), keepdim=True)   # (1, channels, 1)
        
        # Avoid division by zero
        std = torch.clamp(std, min=1e-8)
        
        self.logger.info(f"Normalization stats computed - mean: {mean.mean().item():.4f}, std: {std.mean().item():.4f}")
        return mean, std

    def apply_normalization(self, features: torch.Tensor, mean: Optional[torch.Tensor], std: Optional[torch.Tensor]) -> torch.Tensor:
        """Apply normalization to features."""
        if mean is None or std is None:
            return features
        return (features - mean.to(features.device)) / std.to(features.device)

    def train_epoch(self, model: nn.Module, train_loader: DataLoader, optimizer: optim.Optimizer, 
                   criterion: nn.Module, mean: Optional[torch.Tensor], std: Optional[torch.Tensor]) -> float:
        """Train for one epoch."""
        model.train()
        total_loss = 0.0
        
        for batch_features, batch_labels in train_loader:
            # Handle multi-modal case
            if isinstance(batch_features, dict):
                features = batch_features.get("eeg", batch_features[list(batch_features.keys())[0]])
            else:
                features = batch_features
                
            features = features.to(self.device)
            labels = batch_labels.to(self.device)
            
            # Apply normalization if enabled
            if self.should_normalize:
                features = self.apply_normalization(features, mean, std)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)

    def evaluate(self, model: nn.Module, data_loader: DataLoader, 
                criterion: nn.Module, mean: Optional[torch.Tensor], std: Optional[torch.Tensor]) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """Evaluate model on data loader."""
        model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch_features, batch_labels in data_loader:
                # Handle multi-modal case
                if isinstance(batch_features, dict):
                    features = batch_features.get("eeg", batch_features[list(batch_features.keys())[0]])
                else:
                    features = batch_features
                    
                features = features.to(self.device)
                labels = batch_labels.to(self.device)
                
                # Apply normalization if enabled
                if self.should_normalize:
                    features = self.apply_normalization(features, mean, std)
                
                # Forward pass
                outputs = model(features)
                loss = criterion(outputs, labels)
                
                # Get predictions
                _, predicted = torch.max(outputs, 1)
                
                total_loss += loss.item()
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(data_loader)
        
        # Debug accuracy calculation
        all_predictions_array = np.array(all_predictions)
        all_labels_array = np.array(all_labels)
        
        if len(all_predictions_array) == 0:
            accuracy = 0.0
        else:
            accuracy = accuracy_score(all_labels_array, all_predictions_array)
            
        # Debug logging (only if verbose_training is enabled)
        if hasattr(self, 'verbose_training') and self.verbose_training and len(all_predictions_array) > 0:
            unique_preds, pred_counts = np.unique(all_predictions_array, return_counts=True)
            unique_labels, label_counts = np.unique(all_labels_array, return_counts=True)
            self.logger.info(f"Predictions distribution: {dict(zip(unique_preds, pred_counts))}")
            self.logger.info(f"Labels distribution: {dict(zip(unique_labels, label_counts))}")
            self.logger.info(f"Sample predictions: {all_predictions_array[:10]}")
            self.logger.info(f"Sample labels: {all_labels_array[:10]}")
            self.logger.info(f"Calculated accuracy: {accuracy}")
        
        return avg_loss, accuracy, all_predictions_array, all_labels_array

    # ------------------------------------------------------------------ #
    # per-session processing
    # ------------------------------------------------------------------ #
    def process_session(self, session: Session) -> Dict[str, Any]:
        sid = session.session_id
        bkt = self.s3_bucket
        self.logger.info(f"Training classifier on query data: {sid}")

        # Set random seeds for reproducibility
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # Construct zarr path
        if sid.endswith('.zarr'):
            zarr_path = f"s3://{bkt}/{self.SOURCE_PREFIX}{sid}"
        else:
            zarr_path = f"s3://{bkt}/{self.SOURCE_PREFIX}{sid}.zarr"
        
        try:
            # Load data using enhanced dataloader
            if self.skip_analysis and self.disable_preloading:
                self.logger.info("Loading data with enhanced dataloader (analysis and preloading disabled for speed)")
            elif self.skip_analysis:
                self.logger.info("Loading data with enhanced dataloader (analysis disabled for speed)")
            elif self.disable_preloading:
                self.logger.info("Loading data with enhanced dataloader (preloading disabled for speed)")
            else:
                self.logger.info("Loading data with enhanced dataloader")
            
            data_result = create_neural_data_loaders(
                zarr_path=zarr_path,
                task_type="classification",
                batch_size=self.batch_size,
                modalities=["eeg"],
                train_ratio=self.train_ratio,
                val_ratio=self.val_ratio,
                test_ratio=self.test_ratio,
                auto_balance=self.auto_balance,
                split_by_session=True,  # Prevent data leakage
                seed=self.seed,
                verbose=not self.skip_analysis,
                analyze_distribution=not self.skip_analysis,
                preload_labels=not self.disable_preloading,
                session_limit=self.session_limit
            )
            
            train_loader = data_result['train_loader']
            val_loader = data_result['val_loader']
            test_loader = data_result['test_loader']
            feature_shape = data_result['feature_shape']
            num_classes = data_result['num_classes']
            class_weights = data_result['class_weights']
            distribution_info = data_result['distribution_info']
            
        except Exception as e:
            self.logger.error(f"Failed to load data from {zarr_path}: {e}", exc_info=True)
            return {"status": "failed", "error_details": str(e),
                    "metadata": {"session_id": sid}}

        # Extract feature dimensions
        if isinstance(feature_shape, dict):
            eeg_shape = feature_shape.get("eeg", feature_shape[list(feature_shape.keys())[0]])
        else:
            eeg_shape = feature_shape
            
        n_channels = eeg_shape[0]
        sequence_length = eeg_shape[1] if len(eeg_shape) > 1 else 1
        
        self.logger.info(f"Model input: {n_channels} channels, {sequence_length} timepoints, {num_classes} classes")

        # Create model
        if self.model_type == "baseline_cnn":
            model = BaselineCNN(n_channels, num_classes, sequence_length)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
            
        model = model.to(self.device)
        
        # Setup training
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        
        # Compute normalization statistics if enabled
        mean, std = None, None
        if self.should_normalize:
            mean, std = self.compute_normalization_stats(train_loader, val_loader, test_loader)

        # Training loop with early stopping
        best_val_accuracy = 0.0
        patience_counter = 0
        train_history = []
        best_model_state = model.state_dict().copy()  # Initialize with current state
        
        self.logger.info(f"Starting training for up to {self.num_epochs} epochs")
        
        for epoch in range(self.num_epochs):
            # Train
            train_loss = self.train_epoch(model, train_loader, optimizer, criterion, mean, std)
            
            # Validate (if validation loader exists)
            if val_loader is not None:
                val_loss, val_accuracy, _, _ = self.evaluate(model, val_loader, criterion, mean, std)
            else:
                # Use training accuracy as proxy when no validation set
                val_loss, val_accuracy, _, _ = self.evaluate(model, train_loader, criterion, mean, std)
                self.logger.info(f"No validation set - using training accuracy as proxy")
            
            # Record history
            train_history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy
            })
            
            self.logger.info(f"Epoch {epoch+1:3d}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_accuracy:.4f}")
            
            # Debug: Log more details if accuracy is suspiciously low
            if val_accuracy < 0.1:
                self.logger.warning(f"Very low validation accuracy detected: {val_accuracy:.6f}")
                if self.verbose_training:
                    self.logger.info("This might indicate a data type or prediction issue - use --verbose-training for more details")
            
            # Early stopping check
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                patience_counter = 0
                # Save best model state
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                
            if patience_counter >= self.early_stopping_patience:
                self.logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Load best model and evaluate on test set
        model.load_state_dict(best_model_state)
        if test_loader is not None:
            test_loss, test_accuracy, test_predictions, test_labels = self.evaluate(
                model, test_loader, criterion, mean, std
            )
        else:
            # Use training set for evaluation if no test set
            test_loss, test_accuracy, test_predictions, test_labels = self.evaluate(
                model, train_loader, criterion, mean, std
            )
            self.logger.info(f"No test set - using training set for final evaluation")
        
        # Compute detailed metrics
        report = classification_report(test_labels, test_predictions, output_dict=True)
        conf_matrix = confusion_matrix(test_labels, test_predictions)
        
        self.logger.info(f"Final test accuracy: {test_accuracy:.4f}")
        self.logger.info(f"Classification report:\n{classification_report(test_labels, test_predictions)}")

        # Prepare results
        results = {
            "status": "success",
            "metadata": {
                "session_id": sid,
                "model_type": self.model_type,
                "query_source": zarr_path,
                "num_classes": num_classes,
                "feature_shape": eeg_shape,
                "training_samples": len(train_loader.dataset),
                "validation_samples": len(val_loader.dataset) if val_loader else 0,
                "test_samples": len(test_loader.dataset) if test_loader else 0,
                "epochs_trained": len(train_history),
                "early_stopped": patience_counter >= self.early_stopping_patience,
                "train_accuracy": train_history[-1]['val_accuracy'] if train_history else 0.0,  # Use val acc as proxy
                "val_accuracy": best_val_accuracy,
                "test_accuracy": test_accuracy,
                "class_distribution": distribution_info['label_counts'],
                "classification_report": report,
                "confusion_matrix": conf_matrix.tolist(),
                "hyperparameters": {
                    "learning_rate": self.learning_rate,
                    "batch_size": self.batch_size,
                    "normalize_features": self.should_normalize,
                    "early_stopping_patience": self.early_stopping_patience,
                }
            },
            "model_state": best_model_state,
            "training_history": train_history,
            "normalization_stats": {
                "mean": mean.cpu().numpy().tolist() if mean is not None else None,
                "std": std.cpu().numpy().tolist() if std is not None else None,
            } if self.should_normalize else None
        }

        # Save results to zarr store
        if not self.dry_run:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_key = f"{self.destination_prefix}baseline_{self.query_name}_{timestamp}.zarr"
            
            try:
                self._save_results_to_zarr(results, output_key)
                results["zarr_stores"] = [output_key]
            except Exception as e:
                self.logger.error(f"Failed to save results to zarr: {e}", exc_info=True)
                return {"status": "failed", "error_details": str(e),
                        "metadata": {"session_id": sid}}

        return results

    def _save_results_to_zarr(self, results: Dict[str, Any], output_key: str) -> None:
        """Save training results to zarr store."""
        self.logger.info(f"Saving results to {output_key}")
        
        # Create zarr store
        store = zarr.open(f"s3://{self.s3_bucket}/{output_key}", mode='w')
        
        # Save metadata as root attributes
        metadata = results["metadata"]
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                store.attrs[key] = value
            else:
                # Convert complex objects to JSON strings
                store.attrs[key] = json.dumps(value, default=str)
        
        # Save model state dict
        model_state = results["model_state"]
        model_group = store.create_group("model")
        for param_name, param_tensor in model_state.items():
            # Convert to numpy and save
            param_array = param_tensor.cpu().numpy()
            model_group.create_dataset(param_name, data=param_array)
        
        # Save training history
        if results["training_history"]:
            history = results["training_history"]
            history_group = store.create_group("history")
            for metric in ['epoch', 'train_loss', 'val_loss', 'val_accuracy']:
                if any(metric in epoch_data for epoch_data in history):
                    values = [epoch_data.get(metric, 0) for epoch_data in history]
                    history_group.create_dataset(metric, data=np.array(values))
        
        # Save normalization stats if available
        if results["normalization_stats"]:
            norm_stats = results["normalization_stats"]
            norm_group = store.create_group("normalization")
            if norm_stats["mean"] is not None:
                norm_group.create_dataset("mean", data=np.array(norm_stats["mean"]))
            if norm_stats["std"] is not None:
                norm_group.create_dataset("std", data=np.array(norm_stats["std"]))
        
        self.logger.info(f"Results saved successfully to {output_key}")


# --------------------------------------------------------------------- #
# CLI entry-point
# --------------------------------------------------------------------- #
if __name__ == "__main__":
    BaselineClassifierTransform.run_from_command_line()


# --------------------------------------------------------------------- #
# Example usage for testing
# --------------------------------------------------------------------- #
def test_classifier_training():
    """
    Simple test function to demonstrate classifier usage.
    Run with: python -c "from transforms.t6C_baseline_v0 import test_classifier_training; test_classifier_training()"
    """
    print("Testing baseline classifier...")
    
    # Create classifier with test parameters
    classifier = BaselineClassifierTransform(
        query_name="eye_good",
        learning_rate=0.01,  # Higher LR for faster testing
        num_epochs=3,        # Just a few epochs for testing
        batch_size=16,       # Smaller batch for testing
        early_stopping_patience=2,
        normalize_features=True,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        dry_run=True,        # Don't actually save results
        verbose=True
    )
    
    print(f"Classifier initialized with device: {classifier.device}")
    print(f"Looking for query data matching: {classifier.query_name}")
    
    # Find available sessions
    try:
        sessions = classifier.find_sessions()
        print(f"Found {len(sessions)} sessions: {sessions}")
        
        if sessions:
            # Test on first session
            session_id = sessions[0]
            print(f"\nTesting on session: {session_id}")
            
            result = classifier.process_item(session_id)
            
            print(f"\nTraining completed!")
            print(f"Status: {result.get('status', 'unknown')}")
            if result.get('status') == 'success':
                metadata = result.get('metadata', {})
                test_acc = metadata.get('test_accuracy', 0)
                if isinstance(test_acc, (int, float)):
                    print(f"Test accuracy: {test_acc:.3f}")
                else:
                    print(f"Test accuracy: {test_acc}")
                print(f"Classes: {metadata.get('num_classes', 'N/A')}")
                print(f"Training samples: {metadata.get('training_samples', 'N/A')}")
            else:
                print(f"Error: {result.get('error_details', 'Unknown error')}")
        else:
            print("No sessions found - make sure you've run the query transform first")
            
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__" and len(sys.argv) == 1:
    # If run without arguments, show help
    print(__doc__)
    print("\nFor testing, run:")
    print("python -c \"from transforms.t6C_baseline_v0 import test_classifier_training; test_classifier_training()\"")