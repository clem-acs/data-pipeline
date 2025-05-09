#!/usr/bin/env python3
"""
EEG Quality Analysis Script for gTec Data (Enhanced with FOOOF-based periodic analysis)

This script analyzes EEG data from gTec devices stored in HDF5 files. It:
1. Reads EEG data from the specified H5 file.
2. Applies basic preprocessing (60 Hz notch filter, bandpass filter).
3. Segments the data into 5 equal-length time windows.
4. Uses robust measures to detect and exclude flat/broken channels.
5. Performs spectral analysis (Welch’s method).
6. Fits the power spectrum using FOOOF to separate periodic and aperiodic components.
7. Prints detailed channel-by-channel metrics, including:
   - Raw PSD values
   - Aperiodic component fit
   - Periodic PSD (aperiodic-subtracted)
   - Band-wise periodic power
   - Additional time-segmented stats
8. Produces diagnostic plots to facilitate quality checks.

Usage:
    python eeg-period.py --file "path_to_your_file.h5" [optional arguments]

Requires:
    - numpy
    - scipy
    - matplotlib
    - h5py
    - fooof (pip install fooof)
      (NOTE: 'fooof' is deprecated; new projects can switch to 'specparam'.)
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import re
import random
from scipy import signal
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap

# Import fooof for aperiodic/periodic spectral decomposition
try:
    from fooof import FOOOF
except ImportError:
    print("Warning: The 'fooof' package is not installed. Install via 'pip install fooof' if you want aperiodic subtraction.")
    FOOOF = None

###############################################################################
# Helper Functions
###############################################################################

def parse_h5list_file(h5list_path):
    """
    Parse a 'h5list.txt' file to extract H5 file info.
    Returns a dict mapping filename to shape info, if available.
    """
    file_info = {}
    current_file = None

    with open(h5list_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        # Check for file header
        file_match = re.search(r'## File: (.+\.h5)', line)
        if file_match:
            current_file = file_match.group(1)
            file_info[current_file] = {}
            continue

        # Check for EEG shape
        shape_match = re.search(r'- eeg/frames: \((.+)\)', line)
        if shape_match and current_file:
            shape_str = shape_match.group(1)
            shape_tuple = tuple(map(int, shape_str.split(', ')))
            file_info[current_file]['eeg_shape'] = shape_tuple

    return file_info

def find_h5_files(data_dir):
    """
    Recursively find all .h5 files in the specified directory.
    Returns a list of file paths.
    """
    h5_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".h5"):
                h5_files.append(os.path.join(root, file))
    return h5_files

def robust_channel_detection(eeg_data, std_threshold=0.5, flat_threshold=0.1):
    """
    Identify which channels are active (not disconnected or broken).

    - Clip extreme outliers to reduce huge artifact impact on STD.
    - Compute std after clipping.
    - Mark channels with robust std below 'std_threshold' as likely inactive.
    - Also check 'flat_threshold' for truly flat signals.

    Args:
        eeg_data (ndarray): [channels, samples]
        std_threshold (float): Lower bound on robust std to consider channel active
        flat_threshold (float): Even more stringent threshold for truly flat signals

    Returns:
        active_channels (list): Indices of channels considered active
    """
    n_channels, _ = eeg_data.shape
    data_copy = np.copy(eeg_data)

    # Clip outliers at +/- 5 SD to mitigate single large spikes
    for ch in range(n_channels):
        ch_mean = np.mean(data_copy[ch])
        ch_std = np.std(data_copy[ch])
        clip_min = ch_mean - 5 * ch_std
        clip_max = ch_mean + 5 * ch_std
        data_copy[ch] = np.clip(data_copy[ch], clip_min, clip_max)

    std_per_channel = np.std(data_copy, axis=1)

    # Identify channels above the std_threshold
    active_mask = (std_per_channel > std_threshold)
    # Mark channels that are basically flat
    flat_mask = (std_per_channel < flat_threshold)

    active_channels = []
    for ch_idx in range(n_channels):
        if flat_mask[ch_idx]:
            continue  # This channel is truly flat
        if active_mask[ch_idx]:
            active_channels.append(ch_idx)

    # For typical gTec systems, first 22 are EEG
    eeg_channels = [ch for ch in active_channels if ch < 22]

    if len(eeg_channels) == 0:
        print("Warning: No active channels detected based on robust criteria. Using first 22 as fallback.")
        return list(range(min(22, n_channels)))

    print(f"Detected {len(eeg_channels)} active EEG channels out of 22 possible.")
    return eeg_channels

def apply_notch_filter(data, sample_rate, notch_freq=60.0, quality_factor=30.0):
    """
    Apply a 60 Hz notch filter to reduce power line interference.
    """
    b, a = signal.iirnotch(notch_freq, quality_factor, sample_rate)
    filtered_data = signal.filtfilt(b, a, data, axis=1)
    return filtered_data

def apply_bandpass_filter(data, sample_rate, low_freq=5.0, high_freq=45.0, order=4):
    """
    Apply a Butterworth bandpass filter to keep relevant EEG frequency range.
    """
    nyq = 0.5 * sample_rate
    low = low_freq / nyq
    high = high_freq / nyq

    b, a = signal.butter(order, [low, high], btype='band')
    filtered_data = signal.filtfilt(b, a, data, axis=1)
    return filtered_data

def segment_data(eeg_data, n_segments=5):
    """
    Split the EEG data into 'n_segments' equal-length segments (along time axis).

    Args:
        eeg_data (ndarray): shape [channels, samples]
        n_segments (int): Number of segments to split into

    Returns:
        segments (list of ndarrays): each [channels, seg_length]
    """
    n_channels, n_samples = eeg_data.shape
    seg_length = n_samples // n_segments
    segments = []
    for i in range(n_segments):
        start = i * seg_length
        if i < (n_segments - 1):
            end = (i+1) * seg_length
        else:
            end = n_samples  # last segment includes remainder
        segments.append(eeg_data[:, start:end])
    return segments

def compute_fooof_psd(freqs, psd, freq_range=(1, 50), fooof_settings=None):
    """
    Fit FOOOF model to PSD data and return aperiodic fit, periodic component, etc.

    NOTE: This function now returns 'fooof_freqs' which matches the length of
    'periodic_psd', so they're guaranteed to align. That way we avoid shape mismatch.

    Args:
        freqs (1D array): Frequencies (Hz) for the *full* PSD.
        psd (1D array): Power spectral density (full freq range).
        freq_range (tuple): Lower and upper frequency range for fitting.
        fooof_settings (dict): Settings passed to FOOOF, e.g. {"max_n_peaks": 6}

    Returns:
        fm (FOOOF object) or None
        ap_fit (1D array) or None
        periodic_psd (1D array) or None
        fooof_freqs (1D array) or None
    """
    if FOOOF is None:
        return None, None, None, None  # FOOOF not installed

    if fooof_settings is None:
        fooof_settings = {}

    fm = FOOOF(**fooof_settings)

    # Slice PSD to the specified range
    mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    freqs_in = freqs[mask]
    psd_in = psd[mask]

    # Replace any invalid/zero values (log transform would cause -inf)
    psd_in = np.nan_to_num(psd_in, nan=1e-20, posinf=1e-20, neginf=1e-20)
    psd_in[psd_in <= 0] = 1e-20

    # Fit the FOOOF model
    fm.fit(freqs_in, psd_in)

    # Recreate the full model fit (aperiodic + peaks)
    full_model = fm.fooofed_spectrum_
    ap_fit = fm._ap_fit
    periodic_psd = full_model - ap_fit

    # Return the "frequency axis" that matches the (periodic) PSD
    return fm, ap_fit, periodic_psd, freqs_in  # Key change: also return 'freqs_in'

def compute_band_power(freqs, psd, bands):
    """
    Compute power in each frequency band from PSD.

    Args:
        freqs (1D array): Frequency axis *matching* the psd length
        psd (1D array): PSD values
        bands (dict): e.g. {
            "delta": (1,4),
            "theta": (4,8),
            ...
        }

    Returns:
        band_power_dict (dict): {band_name: power_value, ...}
    """
    band_power_dict = {}
    for band_name, (low, high) in bands.items():
        mask = (freqs >= low) & (freqs < high)
        band_power_dict[band_name] = np.sum(psd[mask])
    return band_power_dict

def analyze_eeg_segments(eeg_data, sample_rate, active_channels, segment_index, bands):
    """
    Analyze one EEG segment for each active channel:
      - Compute Welch PSD (full range)
      - Compute FOOOF aperiodic + periodic (1-50 Hz by default)
      - Compute band powers (from the periodic portion, 1-50 Hz)
      - Print results
      - Return a structured dictionary of results

    Args:
        eeg_data (ndarray): shape [channels, samples], one segment
        sample_rate (float): sampling frequency
        active_channels (list): indices of channels considered "active"
        segment_index (int): which segment index we are analyzing
        bands (dict): band definitions

    Returns:
        results (dict):
            {
               channel_idx: {
                  "raw_psd": [...],
                  "raw_freqs": [...],
                  "fooof_freqs": [...],
                  "aperiodic_psd": [...],
                  "periodic_psd": [...],
                  "band_powers": {...}
               },
               ...
            }
    """
    results = {}
    seg_length = eeg_data.shape[1]
    print(f"\n--- Analyzing Segment {segment_index+1} / (length: {seg_length} samples) ---")

    for ch in active_channels:
        ch_data = eeg_data[ch, :]

        # Welch PSD (full range)
        freqs_full, psd_full = signal.welch(ch_data, fs=sample_rate,
                                           nperseg=min(4*sample_rate, seg_length))

        # Use FOOOF to split into aperiodic & periodic (1-50 Hz)
        fm, ap_fit, periodic_psd, fooof_freqs = compute_fooof_psd(freqs_full, psd_full,
                                                                  freq_range=(1, 50),
                                                                  fooof_settings=None)

        if (fm is not None) and (ap_fit is not None) and (periodic_psd is not None) and (fooof_freqs is not None):
            # Ensure no negative periodic values
            periodic_psd[periodic_psd < 0] = 0.0
            # Compute band powers from the *periodic* portion, using fooof_freqs
            band_power_dict = compute_band_power(fooof_freqs, periodic_psd, bands)
        else:
            # FOOOF not available or failed
            # Fallback: compute band power from the full PSD
            fooof_freqs = freqs_full
            ap_fit = np.zeros_like(psd_full)
            periodic_psd = psd_full
            band_power_dict = compute_band_power(freqs_full, psd_full, bands)

        # Print out info
        print(f" Channel {ch} - Segment {segment_index+1}")
        print(f"   Raw PSD shape: {psd_full.shape}, Freq range: {freqs_full[0]:.1f}-{freqs_full[-1]:.1f} Hz")
        if fm is not None:
            print(f"   Aperiodic Params: {fm.aperiodic_params_}")
            print(f"   # of peaks found: {len(fm.gaussian_params_)}")
        print("   Band Power (Periodic):")
        for bn, val in band_power_dict.items():
            print(f"     {bn}: {val:.4f}")
        print("")

        # Store in results dict
        results[ch] = {
            "raw_freqs": freqs_full,
            "raw_psd": psd_full,
            "fooof_freqs": fooof_freqs,   # Key change: store the freq array used by FOOOF
            "aperiodic_psd": ap_fit,
            "periodic_psd": periodic_psd,
            "band_powers": band_power_dict
        }

    return results

def create_segment_plots(all_segment_results, eeg_data, active_channels, sample_rate, filename, bands):
    """
    Create multi-segment plots summarizing:
      1) Time series for each segment (overlay up to 5 channels)
      2) PSD (raw vs. periodic) for each segment, for up to 5 channels
      3) Summaries of band powers per segment

    Args:
        all_segment_results (list): list of segment result dicts
        eeg_data (ndarray): full EEG data [channels, samples]
        active_channels (list): indices of channels to show
        sample_rate (float)
        filename (str): name of H5 file
        bands (dict): e.g. {"delta": (1,4), "theta": (4,8), ...}
    """
    if len(active_channels) == 0:
        print("No active channels to plot.")
        return

    n_segments = len(all_segment_results)
    display_channels = active_channels[:min(5, len(active_channels))]

    # Prepare figure
    plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(3, n_segments, height_ratios=[1, 1, 1.5])
    plt.suptitle(f'EEG Multi-Segment Analysis - {os.path.basename(filename)}', fontsize=16, y=0.98)

    # 1) Time-series subplots, one per segment
    total_samples = eeg_data.shape[1]
    seg_length = total_samples // n_segments
    seconds_to_show = 5
    samples_to_show = int(seconds_to_show * sample_rate)

    for seg_i in range(n_segments):
        ax_ts = plt.subplot(gs[0, seg_i])
        start = seg_i * seg_length
        end = min(start + seg_length, total_samples)
        for i, ch in enumerate(display_channels):
            data_slice = eeg_data[ch, start:end]
            short_slice = data_slice[:samples_to_show]
            ax_ts.plot(np.arange(len(short_slice)) / sample_rate,
                       short_slice, label=f'Ch {ch}')
        ax_ts.set_title(f'Segment {seg_i+1} (first {seconds_to_show}s)', fontsize=10)
        ax_ts.set_xlabel('Time (s)', fontsize=8)
        if seg_i == 0:
            ax_ts.set_ylabel('Amplitude', fontsize=8)
        ax_ts.grid(True)
        if seg_i == n_segments - 1:
            ax_ts.legend(fontsize=6)

    # 2) PSD plots (raw vs. periodic) for each segment
    for seg_i, seg_res in enumerate(all_segment_results):
        ax_psd = plt.subplot(gs[1, seg_i])
        ax_psd.set_title(f'Segment {seg_i+1} PSD', fontsize=10)
        ax_psd.set_xlabel('Frequency (Hz)', fontsize=8)
        ax_psd.set_ylabel('Power', fontsize=8)
        ax_psd.grid(True)

        for i, ch in enumerate(display_channels):
            ch_res = seg_res.get(ch, None)
            if ch_res is None:
                continue
            freqs_raw = ch_res["raw_freqs"]
            raw_psd = ch_res["raw_psd"]
            fooof_freqs = ch_res["fooof_freqs"]
            periodic_psd = ch_res["periodic_psd"]

            # Plot full PSD
            ax_psd.semilogy(freqs_raw, raw_psd, label=f'Ch{ch} Raw', alpha=0.4)
            # Plot periodic PSD (only in 1-50 Hz range typically)
            ax_psd.semilogy(fooof_freqs, periodic_psd, label=f'Ch{ch} Periodic', alpha=0.8)

        if seg_i == n_segments - 1:
            ax_psd.legend(fontsize=6)

    # 3) Summaries of band powers (periodic) across segments for the displayed channels
    band_names = list(bands.keys())
    n_bands = len(band_names)

    avg_band_powers_by_segment = []
    for seg_i, seg_res in enumerate(all_segment_results):
        cum_power = np.zeros(n_bands)
        count = 0
        for ch in display_channels:
            ch_res = seg_res.get(ch, None)
            if ch_res is not None:
                bp = ch_res["band_powers"]
                for b_i, b_name in enumerate(band_names):
                    cum_power[b_i] += bp[b_name]
                count += 1
        if count > 0:
            avg_band_powers_by_segment.append(cum_power / count)
        else:
            avg_band_powers_by_segment.append(np.zeros(n_bands))

    avg_band_powers_by_segment = np.array(avg_band_powers_by_segment)

    ax_bands = plt.subplot(gs[2, :])
    x = np.arange(n_segments)
    width = 0.1
    for b_i, b_name in enumerate(band_names):
        ax_bands.bar(x + b_i * width - (n_bands * width / 2),
                     avg_band_powers_by_segment[:, b_i],
                     width,
                     label=b_name)

    ax_bands.set_title('Average Periodic Band Power Across Displayed Channels', fontsize=12)
    ax_bands.set_xticks(x)
    ax_bands.set_xticklabels([f'Seg {i+1}' for i in range(n_segments)])
    ax_bands.set_xlabel('Segment', fontsize=10)
    ax_bands.set_ylabel('Power (arbitrary units)', fontsize=10)
    ax_bands.legend()
    ax_bands.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plot_filename = "eeg_segment_analysis.png"
    plt.savefig(plot_filename, dpi=150)
    print(f"Segment analysis plot saved to {plot_filename}")
    plt.close()

def analyze_gtec_eeg_data(h5_file_path, duration_sec=300, exclude_channels=None,
                          low_freq=5.0, high_freq=45.0):
    """
    Main logic to:
      - Load data from H5
      - Reshape to [channels, samples]
      - Possibly limit to 'duration_sec'
      - Filter, detect active channels
      - Segment into 5 windows
      - For each segment, do PSD, FOOOF, etc.
      - Print results
      - Produce plots
    """
    sample_rate = 500  # gTec typical
    print(f"Analyzing file: {os.path.basename(h5_file_path)}")

    # Bands of interest
    bands = {
        "delta": (1, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
        "gamma": (30, 45)
    }

    with h5py.File(h5_file_path, 'r') as f:
        if 'devices' not in f or 'eeg' not in f['devices'] or 'frames_data' not in f['devices']['eeg']:
            print("No EEG device data found in the file.")
            return

        eeg_group = f['devices']['eeg']
        frames_data = eeg_group['frames_data'][:]
        print(f"Original frames_data shape: {frames_data.shape} (chunks, channels, samples_per_chunk)")

        # Reshape to [channels, total_frames]
        total_frames = frames_data.shape[0] * frames_data.shape[2]
        n_channels = frames_data.shape[1]
        reshaped_data = np.zeros((n_channels, total_frames), dtype=np.float32)

        idx = 0
        for chunk_i in range(frames_data.shape[0]):
            chunk_len = frames_data.shape[2]
            for ch_i in range(n_channels):
                reshaped_data[ch_i, idx:idx+chunk_len] = frames_data[chunk_i, ch_i, :]
            idx += chunk_len

        print(f"Reshaped data to: {reshaped_data.shape} (channels, samples)")

        # Possibly limit the duration
        frames_needed = duration_sec * sample_rate
        if frames_needed < reshaped_data.shape[1]:
            reshaped_data = reshaped_data[:, :frames_needed]
            print(f"Limiting to first {frames_needed} frames ({duration_sec}s).")
        else:
            print(f"Data shorter than requested duration. Using all {reshaped_data.shape[1]} frames.")

        # Exclude user-specified channels
        if exclude_channels is not None:
            print(f"Excluding channels: {exclude_channels}")
            for ch_ex in exclude_channels:
                if 0 <= ch_ex < n_channels:
                    reshaped_data[ch_ex, :] = 0.0

        # Apply filters
        reshaped_data = apply_notch_filter(reshaped_data, sample_rate, notch_freq=60.0, quality_factor=30.0)
        reshaped_data = apply_bandpass_filter(reshaped_data, sample_rate, low_freq=low_freq, high_freq=high_freq)

        # Detect active channels
        active_channels = robust_channel_detection(reshaped_data, std_threshold=0.5, flat_threshold=0.1)
        print(f"Active Channels: {active_channels}")

        # Segment data into 5 pieces
        n_segments = 5
        segments = segment_data(reshaped_data, n_segments=n_segments)

        # Analyze each segment
        all_segment_results = []
        for seg_i, seg_data in enumerate(segments):
            seg_results = analyze_eeg_segments(seg_data, sample_rate, active_channels, seg_i, bands)
            all_segment_results.append(seg_results)

        # Compute correlation across entire dataset (active channels)
        print("\n=== Channel Correlation (Whole Dataset) ===")
        active_data_all = reshaped_data[active_channels, :]
        corr_matrix = np.corrcoef(active_data_all)
        # Check for very high correlation pairs
        high_corr_pairs = []
        for i in range(len(active_channels)):
            for j in range(i+1, len(active_channels)):
                if corr_matrix[i, j] > 0.95:
                    high_corr_pairs.append((active_channels[i], active_channels[j], corr_matrix[i, j]))
        if high_corr_pairs:
            print("High correlation detected between channels:")
            for i_ch, j_ch, val in high_corr_pairs:
                print(f"  Channels {i_ch} and {j_ch}: corr={val:.3f}")
        else:
            print("No unusually high correlations > 0.95 between active channels.")

        # Stationarity check (rough)
        print("\n=== Stationarity Check (Whole Data, 3 sub-segments) ===")
        total_samples = reshaped_data.shape[1]
        if total_samples >= 3:
            seg_length_stat = total_samples // 3
            s1 = active_data_all[:, :seg_length_stat]
            s2 = active_data_all[:, seg_length_stat:2*seg_length_stat]
            s3 = active_data_all[:, 2*seg_length_stat:3*seg_length_stat]
            seg_means = [np.mean(s, axis=1) for s in [s1, s2, s3]]
            seg_stds = [np.std(s, axis=1) for s in [s1, s2, s3]]
            mean_diffs = [
                abs(seg_means[0] - seg_means[1]).mean(),
                abs(seg_means[1] - seg_means[2]).mean()
            ]
            std_diffs = [
                abs(seg_stds[0] - seg_stds[1]).mean(),
                abs(seg_stds[1] - seg_stds[2]).mean()
            ]
            mean_stability = np.mean(mean_diffs)
            std_stability = np.mean(std_diffs)
            print(f"Mean stability (lower is better): {mean_stability:.3f}")
            print(f"Std stability (lower is better): {std_stability:.3f}")
            if mean_stability > 10 or std_stability > 5:
                print("Warning: Potential non-stationarity (large variation between sub-segments).")
        else:
            print("Not enough data for stationarity check.")

        # Create multi-segment summary plots
        create_segment_plots(all_segment_results, reshaped_data,
                             active_channels, sample_rate, h5_file_path, bands)

def main():
    parser = argparse.ArgumentParser(description='Analyze gTec EEG data (Enhanced with FOOOF).')
    parser.add_argument('input_file', nargs='?', type=str, help='Specific H5 file (positional argument).')
    parser.add_argument('--file', dest='file_arg', type=str, help='Specific H5 file to analyze')
    parser.add_argument('--dir', type=str, default='./data',
                        help='Directory containing H5 files (default: ./data)')
    parser.add_argument('--duration', type=int, default=300,
                        help='Max duration in seconds to analyze (default: 300)')
    parser.add_argument('--list-files', action='store_true',
                        help='Just list available H5 files without analysis')
    parser.add_argument('--exclude-channels', type=str,
                        help='Comma-separated list of channel indices to exclude from analysis/plotting (e.g., "8,10,12")')
    parser.add_argument('--low-freq', type=float, default=5.0,
                        help='Lower cutoff frequency for bandpass filter in Hz (default: 5.0)')
    parser.add_argument('--high-freq', type=float, default=45.0,
                        help='Upper cutoff frequency for bandpass filter in Hz (default: 45.0)')
    args = parser.parse_args()

    # Determine file to analyze
    file_to_analyze = None
    if args.input_file:
        file_to_analyze = args.input_file
        print(f"Using file from positional argument: {file_to_analyze}")
    elif args.file_arg:
        file_to_analyze = args.file_arg
        print(f"Using file from --file argument: {file_to_analyze}")

    # Parse h5list.txt (if exists)
    h5list_path = os.path.join(args.dir, 'h5list.txt')
    if os.path.exists(h5list_path):
        print(f"Found h5list.txt in {args.dir}, parsing file information...")
        _ = parse_h5list_file(h5list_path)  # not strictly used in this script

    # Find all .h5 files
    h5_files = find_h5_files(args.dir)
    print(f"Found {len(h5_files)} H5 files in {args.dir}.")

    # If --list-files, just list them and exit
    if args.list_files:
        for i, fpath in enumerate(h5_files):
            print(f"{i+1}. {fpath}")
        return

    # Excluded channels
    exclude_channels = None
    if args.exclude_channels:
        try:
            exclude_channels = [int(ch.strip()) for ch in args.exclude_channels.split(',')]
            print(f"Excluding channels: {exclude_channels}")
        except ValueError:
            print("Warning: Invalid --exclude-channels format. Use comma-separated integers.")

    # Proceed to analyze the requested file
    if file_to_analyze:
        if os.path.isfile(file_to_analyze):
            # Full path is valid
            analyze_gtec_eeg_data(file_to_analyze, duration_sec=args.duration,
                                  exclude_channels=exclude_channels,
                                  low_freq=args.low_freq, high_freq=args.high_freq)
        else:
            # Maybe user gave a partial name that exists in the directory
            basename = os.path.basename(file_to_analyze)
            matching = [f for f in h5_files if basename in f]
            if matching:
                analyze_gtec_eeg_data(matching[0], duration_sec=args.duration,
                                      exclude_channels=exclude_channels,
                                      low_freq=args.low_freq, high_freq=args.high_freq)
            else:
                print(f"Could not find file: {file_to_analyze}")
    else:
        # If no file specified, analyze the first file in the list (if any)
        if len(h5_files) > 0:
            print(f"No file specified; analyzing first found file: {h5_files[0]}")
            analyze_gtec_eeg_data(h5_files[0], duration_sec=args.duration,
                                  exclude_channels=exclude_channels,
                                  low_freq=args.low_freq, high_freq=args.high_freq)
        else:
            print("No H5 files found to analyze.")

if __name__ == "__main__":
    main()
