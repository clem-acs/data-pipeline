"""
baseline_logreg_eeg.py
----------------------
Quick logistic-regression baseline on windowed EEG data.

• Loads a query-stage Zarr store via utils.create_data_loaders
• Keeps only windows whose label ∈ {0,1}
• Converts every window -> per-channel mean power in a Welch PSD
• Trains sklearn.linear_model.LogisticRegression (class_weight='balanced')
• Prints balanced-accuracy and a confusion matrix for train / val / test
"""

# --------------------------------------------------------------------- #
#  Imports
# --------------------------------------------------------------------- #
import argparse, json, time, datetime, sys
from pathlib import Path
from typing  import List, Tuple, Dict, Any

import numpy        as np
import torch
from torch.utils.data import DataLoader
from scipy.signal   import welch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, balanced_accuracy_score

# Local helpers (identical import style to classify script)
try:
    from .utils import create_data_loaders          # when installed
except ImportError:
    from utils import create_data_loaders           # when run as script


# --------------------------------------------------------------------- #
#  Feature extraction: Welch PSD → band power per channel
# --------------------------------------------------------------------- #
def window_to_bandpower(x: torch.Tensor,
                        sfreq: float,
                        fmin: float,
                        fmax: float) -> np.ndarray:
    """
    x     : Tensor (C, L)  -- one EEG window, already on CPU
    sfreq : sampling rate in Hz
    fmin/fmax : frequency band for integration
    returns  : ndarray shape (C,) – mean power in band per channel
    """
    np_sig = x.numpy()
    # Welch PSD for every channel
    freqs, psd = welch(np_sig, fs=sfreq, axis=-1, nperseg=min(256, np_sig.shape[-1]))
    # Integrate band
    idx = np.logical_and(freqs >= fmin, freqs <= fmax)
    band_power = psd[:, idx].mean(axis=-1)        # mean power in band
    return band_power.astype(np.float32)          # (C,)


# --------------------------------------------------------------------- #
#  Data-loader → feature matrix helper
# --------------------------------------------------------------------- #
def loader_to_numpy(loader: List[Tuple[torch.Tensor, torch.Tensor]],
                    sfreq: float,
                    fmin: float,
                    fmax: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Flattens a (possibly list-ified) DataLoader into X, y numpy arrays.
    Only keeps labels 0 and 1.
    """
    feats, labs = [], []
    for xb, yb in loader:
        xb = xb.cpu()             # (B,C,L)
        yb = yb.cpu()
        # mask labels
        mask = (yb <= 1)
        xb, yb = xb[mask], yb[mask]
        # per-window band-power features
        for win, lab in zip(xb, yb):
            feats.append(window_to_bandpower(win, sfreq, fmin, fmax))
            labs .append(int(lab))
    return np.stack(feats), np.array(labs)


# --------------------------------------------------------------------- #
#  Evaluation helper
# --------------------------------------------------------------------- #
def evaluate(clf: LogisticRegression,
             X: np.ndarray,
             y: np.ndarray,
             label: str):
    pred = clf.predict(X)
    bal_acc = balanced_accuracy_score(y, pred)
    cm = confusion_matrix(y, pred, labels=[0,1])
    acc0 = cm[0,0] / cm[0].sum()  if cm[0].sum() else 0
    acc1 = cm[1,1] / cm[1].sum()  if cm[1].sum() else 0

    print(f"\n=== {label.upper()} RESULTS ===")
    print(f"Balanced-acc : {bal_acc:.3f}")
    print(f"Class 0 acc  : {acc0:.3f}   ({cm[0,0]}/{cm[0].sum()})")
    print(f"Class 1 acc  : {acc1:.3f}   ({cm[1,1]}/{cm[1].sum()})")
    print("Confusion-matrix:")
    print("       | Pred 0 | Pred 1 |")
    print(f"True 0 | {cm[0,0]:6d} | {cm[0,1]:6d} |")
    print(f"True 1 | {cm[1,0]:6d} | {cm[1,1]:6d} |")


# --------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="Logistic-regression baseline on EEG windows")
    parser.add_argument("--query",        default="eye_neural",
                        help="Query Zarr name (without .zarr)")
    parser.add_argument("--s3-bucket",    default="conduit-data-dev")
    parser.add_argument("--batch-size",   type=int, default=64)
    parser.add_argument("--train-ratio",  type=float, default=0.7)
    parser.add_argument("--val-ratio",    type=float, default=0.15)
    parser.add_argument("--test-ratio",   type=float, default=0.15)
    parser.add_argument("--sfreq",        type=float, default=250,
                        help="Sampling rate in Hz")
    parser.add_argument("--fmin",         type=float, default=8.,
                        help="Lower bound of band (Hz)")
    parser.add_argument("--fmax",         type=float, default=13.,
                        help="Upper bound of band (Hz)")
    parser.add_argument("--split-by-session", action="store_true",
                        help="Group splits by session (recommended)")

    args = parser.parse_args()

    # ---------------------------------------------------------------- #
    # 1. Build DataLoaders (same util as classify script)
    # ---------------------------------------------------------------- #
    query_zarr = f"s3://{args.s3_bucket}/processed/queries/{args.query}.zarr"
    tr_loader, va_loader, te_loader = create_data_loaders(
        zarr_path=query_zarr,
        modalities=["eeg"],
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        shuffle=True,
        num_workers=4,
        split_by_session=args.split_by_session,
    )

    # In case util returns torch-DataLoader objects, keep them *as loaders*
    # (do NOT cast to list) so we still get fresh shuffles.
    # ---------------------------------------------------------------- #
    # 2. Convert loaders → 2-D numpy feature matrices
    # ---------------------------------------------------------------- #
    print("Extracting Welch band-power features ...")
    X_train, y_train = loader_to_numpy(tr_loader, args.sfreq, args.fmin, args.fmax)
    X_val,   y_val   = loader_to_numpy(va_loader, args.sfreq, args.fmin, args.fmax)
    X_test,  y_test  = loader_to_numpy(te_loader, args.sfreq, args.fmin, args.fmax)

    print(f"Feature dim  : {X_train.shape[1]} (channels)")
    print(f"Samples      : train {len(y_train)}, val {len(y_val)}, test {len(y_test)}")

    # ---------------------------------------------------------------- #
    # 3. Fit balanced Logistic-Regression
    # ---------------------------------------------------------------- #
    clf = LogisticRegression(max_iter=1000,
                             class_weight='balanced',
                             solver='lbfgs')
    clf.fit(X_train, y_train)

    # ---------------------------------------------------------------- #
    # 4. Evaluate
    # ---------------------------------------------------------------- #
    evaluate(clf, X_train, y_train, "train")
    if len(y_val):
        evaluate(clf, X_val,   y_val,   "val")
    if len(y_test):
        evaluate(clf, X_test,  y_test,  "test")


# --------------------------------------------------------------------- #
if __name__ == "__main__":
    main()
