#!/usr/bin/env python3
"""
enhanced eeg analysis script for gtec data

this script analyzes eeg data from gtec devices stored in hdf5 files.
it provides detailed quality metrics, spectral analysis with aperiodic component 
removal using fooof, and time-segmented analysis.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import random
import os
import argparse
import re
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
from fooof import FOOOF, FOOOFGroup
from fooof.bands import Bands
from fooof.analysis import get_band_peak_fg
import warnings
import seaborn as sns
from pathlib import Path
import datetime

# define standard frequency bands
FREQ_BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 100)
}

def detect_bad_segments(eeg_data, threshold_std=7.0, threshold_range=2000.0):
    """
    detect segments with large artifacts.
    
    args:
        eeg_data: eeg data array [channels, frames]
        threshold_std: standard deviation threshold multiplier
        threshold_range: absolute range threshold
        
    returns:
        boolean mask of same length as frames, true for good segments
    """
    # get data shape
    n_channels, n_frames = eeg_data.shape
    
    # calculate rolling standard deviation (100ms windows)
    window_size = 50  # 100ms at 500hz
    
    # initialize mask
    good_mask = np.ones(n_frames, dtype=bool)
    
    # for each channel
    for ch in range(n_channels):
        # get channel data
        ch_data = eeg_data[ch, :]
        
        # calculate channel standard deviation excluding extreme values
        sorted_data = np.sort(ch_data)
        trim_idx = int(n_frames * 0.05)  # trim 5% from each end
        trimmed_data = sorted_data[trim_idx:-trim_idx]
        ch_std = np.std(trimmed_data)
        
        # mark samples exceeding threshold
        bad_samples = np.abs(ch_data) > (threshold_std * ch_std)
        
        # also check for extreme ranges
        data_range = np.max(ch_data) - np.min(ch_data)
        if data_range > threshold_range:
            # calculate local ranges
            for i in range(0, n_frames - window_size, window_size // 2):
                segment = ch_data[i:i+window_size]
                seg_range = np.max(segment) - np.min(segment)
                if seg_range > threshold_range:
                    bad_samples[i:i+window_size] = True
        
        # update the overall mask
        good_mask = good_mask & ~bad_samples
    
    # extend bad segments slightly to avoid edge artifacts
    extended_bad_mask = ~good_mask
    kernel_size = 50  # 100ms
    extended_bad_mask = np.convolve(extended_bad_mask, np.ones(kernel_size), mode='same') > 0
    good_mask = ~extended_bad_mask
    
    return good_mask

def robust_channel_detection(eeg_data, min_std=0.05, max_std=1000, correlation_threshold=0.98):
    """
    identify active channels using robust methods that handle artifacts.
    
    args:
        eeg_data: eeg data array [channels, frames]
        min_std: minimum standard deviation to consider a channel active
        max_std: maximum standard deviation to consider a channel active (helps filter extreme artifacts)
        correlation_threshold: threshold for detecting duplicated channels
        
    returns:
        list of indices of valid channels, list of bad channels with reasons
    """
    n_channels, _ = eeg_data.shape
    
    # detect bad segments to exclude from statistics
    good_mask = detect_bad_segments(eeg_data)
    
    # calculate robust statistics for each channel
    channel_stats = []
    for ch in range(min(n_channels, 30)):  # consider up to 30 channels (increased from 22)
        ch_data = eeg_data[ch, good_mask]
        
        # if more than 70% of data is bad, consider the whole channel bad (increased from 50%)
        if len(ch_data) < len(good_mask) * 0.3:
            channel_stats.append({
                'channel': ch,
                'std': np.nan,
                'status': 'bad',
                'reason': 'too many artifacts'
            })
            continue
        
        # calculate statistics
        ch_std = np.std(ch_data)
        ch_mean = np.mean(ch_data)
        ch_median = np.median(ch_data)
        ch_range = np.max(ch_data) - np.min(ch_data)
        
        # check if the channel is flat (too low std)
        if ch_std < min_std:
            status = 'bad'
            reason = 'flat signal'
        # check if the channel has extreme values (too high std)
        elif ch_std > max_std:
            status = 'bad'
            reason = 'extreme artifacts'
        # otherwise, the channel is good
        else:
            status = 'good'
            reason = ''
        
        channel_stats.append({
            'channel': ch,
            'std': ch_std,
            'mean': ch_mean,
            'median': ch_median,
            'range': ch_range,
            'status': status,
            'reason': reason
        })
    
    # convert to dataframe for easier analysis
    stats_df = pd.DataFrame(channel_stats)
    
    # find good channels
    good_channels = stats_df[stats_df['status'] == 'good']['channel'].values.tolist()
    
    # now check for highly correlated channels among good channels
    if len(good_channels) > 1:
        # calculate correlation matrix
        good_data = eeg_data[good_channels, :][:, good_mask]
        try:
            corr_matrix = np.corrcoef(good_data)
            
            # find highly correlated pairs - keeping more channels
            duplicate_channels = set()
            for i in range(len(good_channels)):
                for j in range(i+1, len(good_channels)):
                    if abs(corr_matrix[i, j]) > correlation_threshold:
                        # mark the second channel as duplicate (arbitrary choice)
                        duplicate_channels.add(good_channels[j])
            
            # update the dataframe
            for ch in duplicate_channels:
                idx = stats_df[stats_df['channel'] == ch].index[0]
                stats_df.loc[idx, 'status'] = 'bad'
                stats_df.loc[idx, 'reason'] = 'highly correlated with another channel'
            
            # update good channels
            good_channels = stats_df[stats_df['status'] == 'good']['channel'].values.tolist()
        except:
            print("warning: could not compute correlation matrix, possibly due to constant values or nans")
    
    # bad channels with reasons
    bad_channels = stats_df[stats_df['status'] == 'bad'][['channel', 'reason']].values.tolist()
    
    return good_channels, bad_channels

def segment_eeg_data(eeg_data, num_segments=5):
    """
    segment eeg data into equal time chunks.
    
    args:
        eeg_data: eeg data array [channels, frames]
        num_segments: number of segments to create
        
    returns:
        list of data segments, each shaped [channels, frames_per_segment]
    """
    n_channels, n_frames = eeg_data.shape
    
    # calculate segment size (ensure all segments are equal except possibly the last one)
    segment_size = n_frames // num_segments
    
    segments = []
    for i in range(num_segments):
        start_idx = i * segment_size
        end_idx = start_idx + segment_size if i < num_segments - 1 else n_frames
        segment = eeg_data[:, start_idx:end_idx]
        segments.append(segment)
    
    return segments

def analyze_segment_psd(segment, sample_rate, good_channels, freq_range=(0.5, 100)):
    """
    analyze power spectral density for an eeg segment.
    
    args:
        segment: eeg data segment [channels, frames]
        sample_rate: sampling rate in hz
        good_channels: list of good channel indices
        freq_range: tuple with (min_freq, max_freq) to analyze
        
    returns:
        dictionary with psd analysis results
    """
    results = {}
    
    # calculate optimal nfft for desired frequency resolution
    desired_resolution = 0.25  # hz
    nfft = max(256, int(sample_rate / desired_resolution))
    
    # ensure nfft is a power of 2
    nfft = 2 ** int(np.ceil(np.log2(nfft)))
    
    # set nperseg to be less than or equal to nfft and the segment length
    nperseg = min(4096, nfft, segment.shape[1])
    
    for ch in good_channels:
        ch_data = segment[ch, :]
        
        # calculate psd using welch's method
        freqs, psd = signal.welch(ch_data, sample_rate, nperseg=nperseg, 
                                  nfft=nfft, detrend='constant')
        
        # filter to desired frequency range
        freq_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
        psd_filtered = psd[freq_mask]
        freqs_filtered = freqs[freq_mask]
        
        # calculate band powers
        band_powers = {}
        for band_name, (low, high) in FREQ_BANDS.items():
            band_mask = (freqs_filtered >= low) & (freqs_filtered <= high)
            if np.any(band_mask):
                band_power = np.sum(psd_filtered[band_mask])
                band_powers[band_name] = band_power
        
        # calculate alpha peak frequency
        alpha_mask = (freqs_filtered >= 8) & (freqs_filtered <= 13)
        if np.any(alpha_mask):
            alpha_peak_idx = np.argmax(psd_filtered[alpha_mask])
            alpha_peak_freq = freqs_filtered[alpha_mask][alpha_peak_idx]
            alpha_peak_power = psd_filtered[alpha_mask][alpha_peak_idx]
        else:
            alpha_peak_freq = None
            alpha_peak_power = None
        
        # calculate full spectrum peak
        peak_idx = np.argmax(psd_filtered)
        peak_freq = freqs_filtered[peak_idx]
        peak_power = psd_filtered[peak_idx]
        
        # store results
        results[ch] = {
            'freqs': freqs_filtered,
            'psd': psd_filtered,
            'band_powers': band_powers,
            'alpha_peak_freq': alpha_peak_freq,
            'alpha_peak_power': alpha_peak_power,
            'peak_freq': peak_freq,
            'peak_power': peak_power
        }
    
    return results

def analyze_segment_fooof(segment, sample_rate, good_channels, freq_range=(1, 45)):
    """
    analyze eeg data using fooof to separate periodic and aperiodic components.
    
    args:
        segment: eeg data segment [channels, frames]
        sample_rate: sampling rate in hz
        good_channels: list of good channel indices
        freq_range: tuple with (min_freq, max_freq) for fooof analysis
        
    returns:
        dictionary with fooof analysis results
    """
    results = {}
    
    # create frequency bands for peak extraction
    bands = Bands({'delta': [1, 4],
                  'theta': [4, 8],
                  'alpha': [8, 13],
                  'beta': [13, 30],
                  'gamma': [30, 45]})
    
    # process each channel individually to avoid array shape issues
    for ch_idx, ch in enumerate(good_channels):
        try:
            # extract channel data
            ch_data = segment[ch, :]
            
            # calculate psd using welch's method
            nperseg = min(4096, len(ch_data))
            nfft = max(nperseg, 2**int(np.ceil(np.log2(nperseg))))
            
            freqs, psd = signal.welch(ch_data, fs=sample_rate, nperseg=nperseg, nfft=nfft)
            
            # filter to desired frequency range
            freq_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
            freqs_filtered = freqs[freq_mask]
            psd_filtered = psd[freq_mask]
            
            # check for invalid values
            if np.any(np.isnan(psd_filtered)) or np.any(np.isinf(psd_filtered)):
                print(f"warning: channel {ch} has invalid psd values, skipping fooof analysis")
                continue
            
            # ensure data is numpy arrays
            freqs_filtered = np.asarray(freqs_filtered)
            psd_filtered = np.asarray(psd_filtered)
            
            # initialize fooof model for this channel
            fm = FOOOF(peak_width_limits=[1, 8], max_n_peaks=6, 
                      min_peak_height=0.05, peak_threshold=2.0,
                      aperiodic_mode='fixed')
            
            # fit the model
            fm.fit(freqs_filtered, psd_filtered)
            
            # extract band-specific peaks
            peaks = {}
            for band_name, band_range in bands.bands.items():
                # find peaks in this band
                band_peaks = []
                for peak in fm.peak_params_:
                    cf, pw, bw = peak
                    if band_range[0] <= cf <= band_range[1]:
                        band_peaks.append(peak)
                
                # store the highest peak in the band, if any
                if band_peaks:
                    best_peak = max(band_peaks, key=lambda x: x[1])
                    peaks[band_name] = {
                        'cf': best_peak[0],
                        'power': best_peak[1],
                        'bw': best_peak[2]
                    }
                else:
                    peaks[band_name] = None
            
            # calculate the periodic component (peak fit)
            # the periodic component is the difference between the full model fit and the aperiodic fit
            peak_fit = fm.fooofed_spectrum_ - fm._ap_fit
            
            # store results
            results[ch] = {
                'aperiodic_params': fm.aperiodic_params_,
                'peak_params': fm.peak_params_,
                'r_squared': fm.r_squared_,
                'error': fm.error_,
                'peaks_by_band': peaks,
                'freqs': freqs_filtered,
                'power_spectrum': psd_filtered,
                'aperiodic_fit': fm._ap_fit,
                'fooofed_spectrum': fm.fooofed_spectrum_,
                'periodic_component': peak_fit
            }
            
        except Exception as e:
            print(f"warning: failed to analyze channel {ch} with fooof: {str(e)}")
    
    return results

def apply_notch_filter(data, sample_rate, notch_freq=60.0, quality_factor=30.0):
    """
    apply a notch filter to remove specific frequency noise from eeg data.

    args:
        data: eeg data with shape [channels, frames]
        sample_rate: sampling rate in hz
        notch_freq: frequency to remove (default: 60hz)
        quality_factor: quality factor for the notch filter

    returns:
        filtered data with same shape as input
    """
    b, a = signal.iirnotch(notch_freq, quality_factor, sample_rate)
    filtered_data = np.zeros_like(data)

    # apply the filter to each channel
    for ch in range(data.shape[0]):
        filtered_data[ch, :] = signal.filtfilt(b, a, data[ch, :])

    return filtered_data

def apply_bandpass_filter(data, sample_rate, low_freq=0.5, high_freq=45.0, order=4):
    """
    apply a bandpass filter to keep only the relevant eeg frequency range.

    args:
        data: eeg data with shape [channels, frames]
        sample_rate: sampling rate in hz
        low_freq: lower cutoff frequency (hz)
        high_freq: upper cutoff frequency (hz)
        order: filter order

    returns:
        filtered data with same shape as input
    """
    # normalize the frequencies to nyquist frequency (half the sampling rate)
    nyq = 0.5 * sample_rate
    low = low_freq / nyq
    high = high_freq / nyq

    # design butterworth bandpass filter
    b, a = signal.butter(order, [low, high], btype='band')
    filtered_data = np.zeros_like(data)

    # apply the filter to each channel
    for ch in range(data.shape[0]):
        filtered_data[ch, :] = signal.filtfilt(b, a, data[ch, :])

    return filtered_data

def analyze_gtec_eeg_data(h5_file_path, duration_sec=None, exclude_channels=None, 
                         low_freq=0.5, high_freq=45.0, num_segments=5,
                         analyze_full_file=True):
    """
    analyze gtec eeg data from an h5 file with enhanced features.

    args:
        h5_file_path: path to the h5 file
        duration_sec: duration of data segment to analyze (seconds), none for full file
        exclude_channels: list of channel indices to exclude from analysis
        low_freq: lower cutoff frequency (hz) for bandpass filter
        high_freq: upper cutoff frequency (hz) for bandpass filter
        num_segments: number of time segments to divide the data into
        analyze_full_file: if true, analyze the entire file; if false, analyze a random segment
    """
    print(f"\n{'='*50}")
    print(f"analyzing file: {os.path.basename(h5_file_path)}")
    print(f"{'='*50}\n")

    # open the h5 file
    with h5py.File(h5_file_path, 'r') as f:
        # check if eeg device data exists
        if 'devices' in f and 'eeg' in f['devices']:
            eeg_group = f['devices']['eeg']

            # get frames data
            if 'frames_data' in eeg_group:
                frames_data = eeg_group['frames_data'][:]
                print(f"eeg data shape: {frames_data.shape}")

                # calculate number of frames needed for desired duration
                sample_rate = 500  # hz - adjust if your gtec device uses a different rate
                
                # keep data in format [channels, frames]
                # the data is stored as [chunks, channels, samples]
                # we need to reshape to [channels, total_frames]
                total_frames = frames_data.shape[0] * frames_data.shape[2]
                reshaped_data = np.zeros((frames_data.shape[1], total_frames))

                # concatenate frames for each channel
                for ch in range(frames_data.shape[1]):
                    channel_data = []
                    for chunk in range(frames_data.shape[0]):
                        channel_data.append(frames_data[chunk, ch, :])
                    reshaped_data[ch, :] = np.concatenate(channel_data)
                
                print(f"reshaped data: {reshaped_data.shape} - [channels, frames]")
                print(f"total recording duration: {total_frames/sample_rate:.2f} seconds")

                # check if we need to analyze a specific duration or the full file
                if duration_sec is not None and not analyze_full_file:
                    frames_needed = int(duration_sec * sample_rate)
                    
                    # select a random starting point that allows for full duration
                    if total_frames > frames_needed:
                        start_frame = random.randint(0, total_frames - frames_needed - 1)
                        segment = reshaped_data[:, start_frame:start_frame+frames_needed]
                        print(f"analyzing segment from {start_frame/sample_rate:.2f}s to {(start_frame+frames_needed)/sample_rate:.2f}s")
                    else:
                        segment = reshaped_data
                        print(f"file too short for requested duration, analyzing all available data: {total_frames/sample_rate:.2f}s")
                else:
                    segment = reshaped_data
                    print(f"analyzing full recording: {total_frames/sample_rate:.2f}s")

                # apply 60hz notch filter to reduce power line interference
                segment = apply_notch_filter(segment, sample_rate)
                print("applied 60hz notch filter to reduce power line interference")

                # apply bandpass filter to keep only relevant eeg frequencies
                segment = apply_bandpass_filter(segment, sample_rate, low_freq=low_freq, high_freq=high_freq)
                print(f"applied bandpass filter ({low_freq} to {high_freq} hz) to isolate eeg frequencies")

                # perform robust channel detection
                good_channels, bad_channels = robust_channel_detection(segment)
                
                # print channel detection results
                print(f"\nchannel detection results:")
                print(f"  good channels ({len(good_channels)}): {good_channels}")
                print(f"  bad channels ({len(bad_channels)}):")
                for ch, reason in bad_channels:
                    print(f"    channel {ch}: {reason}")
                
                # handle exclude_channels
                if exclude_channels:
                    good_channels = [ch for ch in good_channels if ch not in exclude_channels]
                    print(f"after excluding specified channels: {len(good_channels)} good channels remaining")
                
                if len(good_channels) == 0:
                    print("no good channels detected. cannot proceed with analysis.")
                    return
                
                # begin detailed analysis
                analyze_eeg_data(segment, sample_rate, good_channels, h5_file_path, num_segments)
            else:
                print("no frames_data found in eeg group")
        else:
            print("no eeg device data found in the file")

def analyze_eeg_data(eeg_data, sample_rate, good_channels, filename, num_segments=5):
    """
    perform detailed analysis on eeg data with time segmentation.

    args:
        eeg_data: eeg data with shape [channels, frames]
        sample_rate: sampling rate in hz
        good_channels: list of indices of good channels to analyze
        filename: source filename for plot title
        num_segments: number of time segments to analyze
    """
    # basic signal checks
    print("\n=== basic signal information ===")
    print(f"shape: {eeg_data.shape}")
    print(f"duration: {eeg_data.shape[1]/sample_rate:.2f} seconds")
    print(f"analyzing {len(good_channels)} good channels: {good_channels}")
    
    # check for nan or inf values in good channels
    active_data = eeg_data[good_channels, :]
    nan_count = np.isnan(active_data).sum()
    inf_count = np.isinf(active_data).sum()
    
    if nan_count > 0:
        print(f"warning: {nan_count} nan values detected in good channels")
    else:
        print("no nan values in good channels")
        
    if inf_count > 0:
        print(f"warning: {inf_count} inf values detected in good channels")
    else:
        print("no inf values in good channels")
    
    # value range for good channels
    min_val = np.min(active_data)
    max_val = np.max(active_data)
    print(f"value range for good channels: {min_val:.3f} to {max_val:.3f}")
    
    # divide data into time segments
    print(f"\n=== analyzing {num_segments} time segments ===")
    segments = segment_eeg_data(eeg_data, num_segments)
    
    # segment information
    for i, segment in enumerate(segments):
        segment_duration = segment.shape[1] / sample_rate
        start_time = i * segment_duration
        end_time = start_time + segment_duration
        print(f"\nsegment {i+1}: {start_time:.2f}s to {end_time:.2f}s ({segment_duration:.2f}s)")
        
        # analyze segment
        segment_results = analyze_segment(segment, sample_rate, good_channels, i+1)
        
        # print segment results summary
        print_segment_summary(segment_results)
    
    # whole recording analysis
    print("\n=== whole recording analysis ===")
    full_results = analyze_segment(eeg_data, sample_rate, good_channels, 0)
    print_segment_summary(full_results)
    
    # create plots for the whole recording
    print("\n=== creating visualization plots ===")
    create_enhanced_plots(eeg_data, sample_rate, good_channels, full_results, segments, filename)

def analyze_segment(segment, sample_rate, good_channels, segment_idx):
    """
    analyze a single segment of eeg data.
    
    args:
        segment: eeg data segment [channels, frames]
        sample_rate: sampling rate in hz
        good_channels: list of good channel indices
        segment_idx: segment index (0 for whole recording)
        
    returns:
        dictionary with analysis results
    """
    # results dictionary
    results = {
        'segment_idx': segment_idx,
        'duration': segment.shape[1] / sample_rate,
        'psd_results': {},
        'fooof_results': {}
    }
    
    # run psd analysis
    psd_results = analyze_segment_psd(segment, sample_rate, good_channels)
    results['psd_results'] = psd_results
    
    # run fooof analysis
    try:
        fooof_results = analyze_segment_fooof(segment, sample_rate, good_channels)
        results['fooof_results'] = fooof_results
    except Exception as e:
        print(f"error in fooof analysis for segment {segment_idx}: {e}")
        results['fooof_results'] = {}
    
    return results

def print_segment_summary(results):
    """
    print a summary of segment analysis results.
    
    args:
        results: dictionary with segment analysis results
    """
    segment_idx = results['segment_idx']
    duration = results['duration']
    psd_results = results['psd_results']
    fooof_results = results['fooof_results']
    
    segment_name = "whole recording" if segment_idx == 0 else f"segment {segment_idx}"
    print(f"\n--- {segment_name} summary (duration: {duration:.2f}s) ---")
    
    # psd summary
    if psd_results:
        print("\npower spectral density analysis:")
        
        # collect data for all channels
        all_band_powers = {band: [] for band in FREQ_BANDS.keys()}
        all_peak_freqs = []
        
        for ch, ch_data in psd_results.items():
            # channel-specific data
            peak_freq = ch_data['peak_freq']
            band_powers = ch_data['band_powers']
            
            # print channel summary
            print(f"  channel {ch}:")
            print(f"    dominant frequency: {peak_freq:.2f} hz")
            
            # print band powers
            for band, power in band_powers.items():
                print(f"    {band.capitalize()} power: {power:.6f}")
                all_band_powers[band].append(power)
            
            # add to overall statistics
            all_peak_freqs.append(peak_freq)
        
        # print summary statistics across channels
        print("\n  channel-average frequency band power:")
        for band, powers in all_band_powers.items():
            if powers:
                print(f"    {band.capitalize()}: {np.mean(powers):.6f} (±{np.std(powers):.6f})")
        
        if all_peak_freqs:
            print(f"  average dominant frequency: {np.mean(all_peak_freqs):.2f} hz (±{np.std(all_peak_freqs):.2f})")
    
    # fooof summary
    if fooof_results:
        print("\nfooof analysis (periodic/aperiodic component separation):")
        
        # collect band peak data across channels
        band_peaks = {band: {'cf': [], 'power': [], 'count': 0} for band in FREQ_BANDS.keys()}
        
        # collect aperiodic parameters
        aperiodic_offsets = []
        aperiodic_exponents = []
        r_squared_values = []
        
        for ch, ch_data in fooof_results.items():
            if ch_data is None:
                continue
                
            # channel-specific data
            aperiodic_params = ch_data['aperiodic_params']
            r_squared = ch_data['r_squared']
            
            # print channel summary
            print(f"  channel {ch}:")
            print(f"    model fit r²: {r_squared:.4f}")
            print(f"    aperiodic component: offset={aperiodic_params[0]:.4f}, exponent={aperiodic_params[1]:.4f}")
            
            # add to overall statistics
            aperiodic_offsets.append(aperiodic_params[0])
            aperiodic_exponents.append(aperiodic_params[1])
            r_squared_values.append(r_squared)
            
            # print peaks by band
            print("    oscillatory peaks by band:")
            peaks_by_band = ch_data['peaks_by_band']
            for band, peak in peaks_by_band.items():
                if peak is not None:
                    print(f"      {band.capitalize()}: {peak['cf']:.2f} hz, power: {peak['power']:.4f}, bandwidth: {peak['bw']:.2f}")
                    # add to band statistics
                    band_peaks[band]['cf'].append(peak['cf'])
                    band_peaks[band]['power'].append(peak['power'])
                    band_peaks[band]['count'] += 1
                else:
                    print(f"      {band.capitalize()}: no significant peak detected")
        
        # print summary statistics across channels
        print("\n  channel-average fooof metrics:")
        if aperiodic_offsets:
            print(f"    aperiodic offset: {np.mean(aperiodic_offsets):.4f} (±{np.std(aperiodic_offsets):.4f})")
        if aperiodic_exponents:
            print(f"    aperiodic exponent: {np.mean(aperiodic_exponents):.4f} (±{np.std(aperiodic_exponents):.4f})")
        if r_squared_values:
            print(f"    model fit r²: {np.mean(r_squared_values):.4f} (±{np.std(r_squared_values):.4f})")
        
        print("\n  channel-average oscillatory peaks:")
        for band, data in band_peaks.items():
            if data['count'] > 0:
                detection_rate = data['count'] / len(fooof_results) * 100
                print(f"    {band.capitalize()}: detected in {detection_rate:.1f}% of channels")
                if data['cf']:
                    print(f"      frequency: {np.mean(data['cf']):.2f} hz (±{np.std(data['cf']):.2f})")
                if data['power']:
                    print(f"      power: {np.mean(data['power']):.4f} (±{np.std(data['power']):.4f})")
            else:
                print(f"    {band.capitalize()}: no peaks detected in any channel")

def create_enhanced_plots(eeg_data, sample_rate, good_channels, full_results, segments, filename):
    """
    create enhanced plots for eeg analysis results.
    
    args:
        eeg_data: full eeg data [channels, frames]
        sample_rate: sampling rate in hz
        good_channels: list of good channel indices
        full_results: analysis results for the full recording
        segments: list of segmented data
        filename: source filename for plot title
    """
    basename = os.path.basename(filename)
    output_dir = f"eeg_analysis_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. overview plot with time series, psds, and channel quality
    create_overview_plot(eeg_data, sample_rate, good_channels, full_results, os.path.join(output_dir, f"{basename}_overview.png"))
    
    # 2. fooof analysis plot
    create_fooof_plot(full_results, good_channels, os.path.join(output_dir, f"{basename}_fooof.png"))
    
    # 3. time segment comparison plot
    create_segment_comparison_plot(segments, sample_rate, good_channels, os.path.join(output_dir, f"{basename}_segments.png"))
    
    # 4. channel correlation plot
    create_channel_correlation_plot(eeg_data, good_channels, os.path.join(output_dir, f"{basename}_correlation.png"))
    
    # 5. channel-by-channel detailed analysis
    for ch in good_channels[:min(5, len(good_channels))]:  # limit to first 5 channels to avoid too many plots
        create_channel_detail_plot(eeg_data, sample_rate, ch, full_results, os.path.join(output_dir, f"{basename}_ch{ch}_detail.png"))
    
    print(f"plots saved to directory: {output_dir}")

def create_overview_plot(eeg_data, sample_rate, good_channels, results, output_file):
    """
    create an overview plot with time series and spectral information.
    
    args:
        eeg_data: eeg data [channels, frames]
        sample_rate: sampling rate in hz
        good_channels: list of good channel indices
        results: analysis results
        output_file: output file path
    """
    plt.figure(figsize=(20, 16))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # set up the layout
    gs = gridspec.GridSpec(4, 2, height_ratios=[1, 1, 1, 1])
    
    # 1. time series plot - show first 10 seconds of data
    ax_time = plt.subplot(gs[0, 0])
    seconds_to_show = 10
    samples_to_show = min(int(seconds_to_show * sample_rate), eeg_data.shape[1])
    
    # select up to 5 channels for clarity
    display_channels = good_channels[:min(5, len(good_channels))]
    colors = plt.cm.viridis(np.linspace(0, 1, len(display_channels)))
    
    for i, ch in enumerate(display_channels):
        ax_time.plot(np.arange(samples_to_show) / sample_rate, 
                    eeg_data[ch, :samples_to_show], 
                    label=f'ch {ch}', 
                    color=colors[i],
                    alpha=0.8)
    
    ax_time.set_title('eeg time series (first 10 seconds)', fontsize=14)
    ax_time.set_xlabel('time (seconds)', fontsize=12)
    ax_time.set_ylabel('amplitude', fontsize=12)
    ax_time.legend(loc='upper right')
    ax_time.grid(True)
    
    # 2. power spectral density plot
    ax_psd = plt.subplot(gs[0, 1])
    
    for i, ch in enumerate(display_channels):
        if ch in results['psd_results']:
            freqs = results['psd_results'][ch]['freqs']
            psd = results['psd_results'][ch]['psd']
            # limit to 50 hz for better visualization
            mask = freqs <= 50
            ax_psd.semilogy(freqs[mask], psd[mask], label=f'ch {ch}', color=colors[i], alpha=0.8)
    
    # add frequency band shading
    band_colors = {
        'delta': '#FFD700',
        'theta': '#FF6347',
        'alpha': '#32CD32',
        'beta': '#1E90FF',
        'gamma': '#9370DB'
    }
    
    for band, (low, high) in FREQ_BANDS.items():
        if high <= 50:  # only show bands up to 50 hz
            ax_psd.axvspan(low, high, color=band_colors[band], alpha=0.2, label=f'{band.capitalize()} ({low} to {high} hz)')
    
    ax_psd.set_title('power spectral density', fontsize=14)
    ax_psd.set_xlabel('frequency (hz)', fontsize=12)
    ax_psd.set_ylabel('power (log scale)', fontsize=12)
    ax_psd.legend(loc='upper right', fontsize=8)
    ax_psd.grid(True)
    
    # 3. relative band power across channels
    ax_bands = plt.subplot(gs[1, 0])
    
    # collect band powers for all channels
    band_powers = {}
    for band in FREQ_BANDS.keys():
        band_powers[band] = []
    
    total_powers = []
    
    for ch in good_channels:
        if ch in results['psd_results']:
            ch_powers = results['psd_results'][ch]['band_powers']
            total_power = sum(ch_powers.values())
            total_powers.append(total_power)
            
            for band, power in ch_powers.items():
                band_powers[band].append(power / total_power if total_power > 0 else 0)
    
    # convert to arrays and calculate mean/std
    for band in band_powers:
        if band_powers[band]:
            band_powers[band] = np.array(band_powers[band])
    
    # plot relative band powers
    band_labels = [band.capitalize() for band in FREQ_BANDS.keys()]
    x = np.arange(len(band_labels))
    
    for i, band in enumerate(FREQ_BANDS.keys()):
        if band_powers[band].size > 0:
            mean_power = np.mean(band_powers[band]) * 100  # convert to percentage
            std_power = np.std(band_powers[band]) * 100
            ax_bands.bar(i, mean_power, yerr=std_power, color=band_colors[band], 
                        capsize=5, label=band.capitalize())
            ax_bands.text(i, mean_power + std_power + 2, f"{mean_power:.1f}%", 
                        ha='center', va='bottom', fontsize=10)
    
    ax_bands.set_title('relative band power (% of total power)', fontsize=14)
    ax_bands.set_ylabel('relative power (%)', fontsize=12)
    ax_bands.set_xticks(x)
    ax_bands.set_xticklabels(band_labels)
    ax_bands.grid(True, axis='y')
    
    # 4. signal quality metrics
    ax_quality = plt.subplot(gs[1, 1])
    
    # calculate quality metrics for each channel
    quality_metrics = []
    channel_labels = []
    
    for ch in good_channels:
        # get the fourier spectrum if available
        snr = 0
        if ch in results['psd_results']:
            # calculate snr as ratio of eeg band power to high frequency noise
            psd_data = results['psd_results'][ch]
            freqs = psd_data['freqs']
            psd = psd_data['psd']
            
            # eeg signal power (0.5 to 30 hz)
            signal_mask = (freqs >= 0.5) & (freqs <= 30)
            signal_power = np.sum(psd[signal_mask])
            
            # noise power (above 50 hz)
            noise_mask = freqs >= 50
            if np.any(noise_mask):
                noise_power = np.sum(psd[noise_mask])
                snr = signal_power / (noise_power + 1e-10)  # avoid division by zero
        
        # calculate other metrics
        ch_data = eeg_data[ch, :]
        std_dev = np.std(ch_data)
        range_val = np.max(ch_data) - np.min(ch_data)
        
        # calculate line noise (if we have psd results)
        line_noise = 0
        if ch in results['psd_results']:
            freqs = results['psd_results'][ch]['freqs']
            psd = results['psd_results'][ch]['psd']
            line_mask_50 = (freqs >= 49) & (freqs <= 51)
            line_mask_60 = (freqs >= 59) & (freqs <= 61)
            line_noise_50 = np.sum(psd[line_mask_50]) if np.any(line_mask_50) else 0
            line_noise_60 = np.sum(psd[line_mask_60]) if np.any(line_mask_60) else 0
            line_noise = max(line_noise_50, line_noise_60)
        
        # add to the list
        quality_metrics.append({
            'channel': ch,
            'snr': snr,
            'std': std_dev,
            'range': range_val,
            'line_noise': line_noise
        })
        channel_labels.append(f"ch {ch}")
    
    # plot the metrics
    if quality_metrics:
        df = pd.DataFrame(quality_metrics)
        
        # normalize metrics for plotting
        if 'snr' in df.columns and not df['snr'].isna().all():
            df['snr_norm'] = df['snr'] / df['snr'].max() * 100 if df['snr'].max() > 0 else 0
        else:
            df['snr_norm'] = 0
            
        if 'std' in df.columns and not df['std'].isna().all():
            df['std_norm'] = df['std'] / df['std'].max() * 100 if df['std'].max() > 0 else 0
        else:
            df['std_norm'] = 0
            
        if 'line_noise' in df.columns and not df['line_noise'].isna().all():
            max_noise = df['line_noise'].max()
            # invert so lower noise is better
            df['noise_norm'] = 100 - (df['line_noise'] / max_noise * 100 if max_noise > 0 else 0)
        else:
            df['noise_norm'] = 100
        
        # create a bar plot for each metric
        bar_width = 0.25
        r1 = np.arange(len(df))
        r2 = [x + bar_width for x in r1]
        r3 = [x + bar_width for x in r2]
        
        ax_quality.bar(r1, df['snr_norm'], width=bar_width, label='snr', color='green', alpha=0.7)
        ax_quality.bar(r2, df['std_norm'], width=bar_width, label='signal variance', color='blue', alpha=0.7)
        ax_quality.bar(r3, df['noise_norm'], width=bar_width, label='line noise (inverted)', color='red', alpha=0.7)
        
        ax_quality.set_ylabel('normalized quality (%)', fontsize=12)
        ax_quality.set_title('signal quality metrics by channel', fontsize=14)
        ax_quality.set_xticks([r + bar_width for r in range(len(df))])
        ax_quality.set_xticklabels(channel_labels)
        ax_quality.set_ylim(0, 110)
        ax_quality.legend()
        ax_quality.grid(True, axis='y')
    
    # 5. time-frequency analysis (spectrogram) for a representative channel
    ax_specgram = plt.subplot(gs[2, :])
    
    if good_channels:
        # use the first good channel for spectrogram
        ch = good_channels[0]
        ch_data = eeg_data[ch, :]
        
        # calculate spectrogram
        nperseg = min(int(2 * sample_rate), len(ch_data))  # 2-second window
        noverlap = nperseg // 2  # 50% overlap
        
        f, t, Sxx = signal.spectrogram(ch_data, fs=sample_rate, nperseg=nperseg, noverlap=noverlap)
        
        # plot spectrogram up to 50 hz
        f_mask = f <= 50
        im = ax_specgram.pcolormesh(t, f[f_mask], 10 * np.log10(Sxx[f_mask, :]), 
                                  shading='gouraud', cmap='viridis')
        
        # add colorbar
        cbar = plt.colorbar(im, ax=ax_specgram)
        cbar.set_label('power/frequency (db/hz)', fontsize=10)
        
        # add frequency band lines
        for band, (low, high) in FREQ_BANDS.items():
            if high <= 50:  # only show bands up to 50 hz
                ax_specgram.axhline(y=low, color=band_colors[band], linestyle='--', alpha=0.8, linewidth=1)
                ax_specgram.axhline(y=high, color=band_colors[band], linestyle='--', alpha=0.8, linewidth=1)
                # add band label
                ax_specgram.text(t[-1]*1.01, (low+high)/2, band.capitalize(), 
                               color=band_colors[band], va='center', ha='left', fontsize=8)
        
        ax_specgram.set_title(f'spectrogram for channel {ch}', fontsize=14)
        ax_specgram.set_xlabel('time (seconds)', fontsize=12)
        ax_specgram.set_ylabel('frequency (hz)', fontsize=12)
    
    # 6. psd comparison with aperiodic component for a representative channel
    ax_fooof = plt.subplot(gs[3, 0])
    
    if good_channels and results['fooof_results']:
        # use the first good channel with fooof results
        for ch in good_channels:
            if ch in results['fooof_results'] and results['fooof_results'][ch] is not None:
                ch_fooof = results['fooof_results'][ch]
                
                # plot original spectrum
                freqs = ch_fooof['freqs']
                power_spectrum = ch_fooof['power_spectrum']
                ax_fooof.semilogy(freqs, power_spectrum, 'k-', label='original spectrum', alpha=0.8)
                
                # plot fooof model fit
                fooofed_spectrum = ch_fooof['fooofed_spectrum']
                ax_fooof.semilogy(freqs, fooofed_spectrum, 'r-', label='fooof model fit', alpha=0.8)
                
                # plot aperiodic component
                aperiodic_fit = ch_fooof['aperiodic_fit']
                ax_fooof.semilogy(freqs, aperiodic_fit, 'b--', label='aperiodic component', alpha=0.8)
                
                # get model fit statistics
                r_squared = ch_fooof['r_squared']
                error = ch_fooof['error']
                
                ax_fooof.set_title(f'fooof analysis for channel {ch} (r² = {r_squared:.4f})', fontsize=14)
                ax_fooof.set_xlabel('frequency (hz)', fontsize=12)
                ax_fooof.set_ylabel('power (log scale)', fontsize=12)
                ax_fooof.legend(loc='upper right', fontsize=10)
                ax_fooof.grid(True, which='both', linestyle='--', alpha=0.5)
                
                # limit view to 50 hz
                ax_fooof.set_xlim(0, 50)
                
                break
    
    # 7. peak frequency distribution
    ax_peaks = plt.subplot(gs[3, 1])
    
    # collect peak frequencies from all channels
    alpha_peaks = []
    peak_channels = []
    
    for ch in good_channels:
        if ch in results['fooof_results'] and results['fooof_results'][ch] is not None:
            fooof_ch = results['fooof_results'][ch]
            if 'peaks_by_band' in fooof_ch and 'alpha' in fooof_ch['peaks_by_band']:
                alpha_peak = fooof_ch['peaks_by_band']['alpha']
                if alpha_peak is not None:
                    alpha_peaks.append(alpha_peak['cf'])
                    peak_channels.append(ch)
    
    if alpha_peaks:
        # create scatter plot of peak frequencies
        ax_peaks.scatter(peak_channels, alpha_peaks, s=80, c='purple', alpha=0.7, edgecolor='k')
        
        # add mean line
        mean_freq = np.mean(alpha_peaks)
        ax_peaks.axhline(y=mean_freq, color='red', linestyle='--', 
                        label=f'mean: {mean_freq:.2f} hz')
        
        # add individual alpha band range
        ax_peaks.axhspan(8, 13, color='green', alpha=0.1, label='alpha band (8 to 13 hz)')
        
        ax_peaks.set_title('alpha peak frequency by channel', fontsize=14)
        ax_peaks.set_xlabel('channel', fontsize=12)
        ax_peaks.set_ylabel('frequency (hz)', fontsize=12)
        ax_peaks.legend(loc='upper right', fontsize=10)
        ax_peaks.grid(True)
        
        # set y-axis limits to focus on alpha range with some padding
        y_min = max(0, min(alpha_peaks) - 2)
        y_max = max(alpha_peaks) + 2
        ax_peaks.set_ylim(y_min, y_max)
    
    # add a title for the entire figure
    plt.suptitle(f'eeg analysis summary - {os.path.basename(output_file)}', fontsize=16, y=0.98)
    
    # adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # save the figure
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"overview plot saved to {output_file}")
    plt.close()

def create_fooof_plot(results, good_channels, output_file):
    """
    create a detailed fooof analysis plot for periodic/aperiodic components.
    
    args:
        results: analysis results
        good_channels: list of good channel indices
        output_file: output file path
    """
    # skip if no fooof results
    if not results['fooof_results']:
        print("no fooof results available for plotting")
        return
    
    # select channels with valid fooof results
    fooof_channels = [ch for ch in good_channels if ch in results['fooof_results'] 
                     and results['fooof_results'][ch] is not None]
    
    if not fooof_channels:
        print("no valid fooof results found for any channel")
        return
    
    # select up to 6 channels to display
    display_channels = fooof_channels[:min(6, len(fooof_channels))]
    
    # create figure
    fig = plt.figure(figsize=(18, 15))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # set up grid
    n_channels = len(display_channels)
    n_rows = (n_channels + 1) // 2  # channels plus summary
    gs = gridspec.GridSpec(n_rows + 1, 2)  # +1 for summary plot
    
    # plot individual channel fooof results
    for i, ch in enumerate(display_channels):
        row = i // 2
        col = i % 2
        
        ax = fig.add_subplot(gs[row, col])
        
        fooof_data = results['fooof_results'][ch]
        
        # plot original spectrum
        freqs = fooof_data['freqs']
        power_spectrum = fooof_data['power_spectrum']
        ax.semilogy(freqs, power_spectrum, 'k-', label='original spectrum', alpha=0.8)
        
        # plot fooof model fit
        fooofed_spectrum = fooof_data['fooofed_spectrum']
        ax.semilogy(freqs, fooofed_spectrum, 'r-', label='fooof model fit', alpha=0.8)
        
        # plot aperiodic component
        aperiodic_fit = fooof_data['aperiodic_fit']
        ax.semilogy(freqs, aperiodic_fit, 'b--', label='aperiodic component', alpha=0.8)
        
        # add detected peaks as vertical spans
        peak_params = fooof_data['peak_params']
        for peak in peak_params:
            cf, pw, bw = peak
            # calculate gaussian width (2*std)
            width = bw / 2
            ax.axvspan(cf - width, cf + width, color='green', alpha=0.2)
            ax.axvline(cf, color='green', linestyle='-', alpha=0.5, linewidth=1)
        
        # add band peaks if available
        if 'peaks_by_band' in fooof_data:
            for band, peak in fooof_data['peaks_by_band'].items():
                if peak is not None:
                    # get band color
                    band_color = {
                        'delta': 'yellow',
                        'theta': 'orange',
                        'alpha': 'green',
                        'beta': 'blue',
                        'gamma': 'purple'
                    }.get(band, 'gray')
                    
                    # add annotation
                    ax.annotate(f"{band.capitalize()}: {peak['cf']:.1f} hz", 
                               xy=(peak['cf'], power_spectrum[np.abs(freqs - peak['cf']).argmin()]),
                               xytext=(peak['cf'] + 1, power_spectrum[np.abs(freqs - peak['cf']).argmin()] * 1.5),
                               arrowprops=dict(facecolor=band_color, shrink=0.05, width=1.5, headwidth=7),
                               fontsize=8, color=band_color)
        
        # get model fit statistics
        r_squared = fooof_data['r_squared']
        error = fooof_data['error']
        aperiodic_params = fooof_data['aperiodic_params']
        
        # add aperiodic parameters text
        ax.text(0.02, 0.98, 
               f"aperiodic:\noffset: {aperiodic_params[0]:.2f}\nexponent: {aperiodic_params[1]:.2f}\nr²: {r_squared:.4f}",
               transform=ax.transAxes, fontsize=10, va='top', 
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_title(f'channel {ch} fooof analysis', fontsize=12)
        ax.set_xlabel('frequency (hz)', fontsize=10)
        ax.set_ylabel('power (log scale)', fontsize=10)
        if i == 0:  # only show legend for first plot
            ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, which='both', linestyle='--', alpha=0.5)
        
        # limit view to improve readability
        ax.set_xlim(0, 45)
    
    # create summary plots for the bottom row
    
    # left: aperiodic parameters across channels
    ax_aperiodic = fig.add_subplot(gs[n_rows, 0])
    
    # collect aperiodic parameters
    offsets = []
    exponents = []
    r_squared_values = []
    channels = []
    
    for ch in fooof_channels:
        fooof_data = results['fooof_results'][ch]
        if fooof_data:
            aperiodic_params = fooof_data['aperiodic_params']
            offsets.append(aperiodic_params[0])
            exponents.append(aperiodic_params[1])
            r_squared_values.append(fooof_data['r_squared'])
            channels.append(ch)
    
    if channels:
        # plot parameters
        x = np.arange(len(channels))
        width = 0.35
        
        ax_aperiodic.bar(x - width/2, offsets, width, label='offset', color='blue', alpha=0.7)
        ax_aperiodic.bar(x + width/2, exponents, width, label='exponent', color='red', alpha=0.7)
        
        # add r² as text
        for i, r2 in enumerate(r_squared_values):
            ax_aperiodic.text(i, max(offsets[i], exponents[i]) + 0.1, f"r²: {r2:.2f}", 
                            ha='center', va='bottom', fontsize=8, rotation=90)
        
        ax_aperiodic.set_title('aperiodic parameters by channel', fontsize=12)
        ax_aperiodic.set_xlabel('channel', fontsize=10)
        ax_aperiodic.set_ylabel('parameter value', fontsize=10)
        ax_aperiodic.set_xticks(x)
        ax_aperiodic.set_xticklabels(channels)
        ax_aperiodic.legend()
        ax_aperiodic.grid(True, axis='y')
    
    # right: peak presence across channels and bands
    ax_peaks = fig.add_subplot(gs[n_rows, 1])
    
    # collect peak information
    bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    peak_presence = np.zeros((len(fooof_channels), len(bands)))
    peak_freqs = np.zeros((len(fooof_channels), len(bands)))
    peak_freqs.fill(np.nan)  # fill with nan for missing peaks
    
    for i, ch in enumerate(fooof_channels):
        fooof_data = results['fooof_results'][ch]
        if fooof_data and 'peaks_by_band' in fooof_data:
            for j, band in enumerate(bands):
                if band in fooof_data['peaks_by_band'] and fooof_data['peaks_by_band'][band] is not None:
                    peak_presence[i, j] = 1
                    peak_freqs[i, j] = fooof_data['peaks_by_band'][band]['cf']
    
    # create heatmap for peak presence
    im = ax_peaks.imshow(peak_presence, cmap='YlGnBu', aspect='auto')
    
    # add center frequency as text
    for i in range(len(fooof_channels)):
        for j in range(len(bands)):
            if not np.isnan(peak_freqs[i, j]):
                ax_peaks.text(j, i, f"{peak_freqs[i, j]:.1f}", ha="center", va="center", 
                             color="black" if peak_presence[i, j] < 0.5 else "white", fontsize=8)
    
    # customize plot
    ax_peaks.set_title('oscillatory peak presence by band and channel', fontsize=12)
    ax_peaks.set_xticks(np.arange(len(bands)))
    ax_peaks.set_yticks(np.arange(len(fooof_channels)))
    ax_peaks.set_xticklabels([b.capitalize() for b in bands])
    ax_peaks.set_yticklabels(fooof_channels)
    ax_peaks.set_xlabel('frequency band', fontsize=10)
    ax_peaks.set_ylabel('channel', fontsize=10)
    
    # add colorbar
    cbar = plt.colorbar(im, ax=ax_peaks, ticks=[0, 1])
    cbar.set_ticklabels(['no peak', 'peak'])
    
    # add a title for the entire figure
    fig.suptitle('fooof analysis - periodic and aperiodic components', fontsize=16, y=0.98)
    
    # adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # save the figure
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"fooof analysis plot saved to {output_file}")
    plt.close()

def create_segment_comparison_plot(segments, sample_rate, good_channels, output_file):
    """
    create a plot comparing analysis across time segments.
    
    args:
        segments: list of eeg data segments [channels, frames]
        sample_rate: sampling rate in hz
        good_channels: list of good channel indices
        output_file: output file path
    """
    if not segments or not good_channels:
        print("no segments or good channels available for comparison plot")
        return
    
    # analyze each segment
    segment_results = []
    for i, segment in enumerate(segments):
        # calculate segment time
        segment_duration = segment.shape[1] / sample_rate
        start_time = i * segment_duration
        end_time = start_time + segment_duration
        
        # analyze psd for the segment
        psd_results = analyze_segment_psd(segment, sample_rate, good_channels)
        
        # store results
        segment_results.append({
            'idx': i + 1,
            'start_time': start_time,
            'end_time': end_time,
            'psd_results': psd_results
        })
    
    # create figure
    plt.figure(figsize=(18, 15))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # set up grid
    gs = gridspec.GridSpec(3, 2)
    
    # 1. band power over time (top left)
    ax_band_time = plt.subplot(gs[0, 0])
    
    # collect band powers across segments and channels
    bands = list(FREQ_BANDS.keys())
    
    # dictionary to store average band power per segment
    avg_band_powers = {band: [] for band in bands}
    segment_times = []
    
    for segment in segment_results:
        # calculate the central time for this segment
        central_time = (segment['start_time'] + segment['end_time']) / 2
        segment_times.append(central_time)
        
        # initialize band power accumulator for this segment
        segment_band_powers = {band: [] for band in bands}
        
        # collect powers from all channels
        for ch in good_channels:
            if ch in segment['psd_results']:
                ch_band_powers = segment['psd_results'][ch]['band_powers']
                for band in bands:
                    if band in ch_band_powers:
                        segment_band_powers[band].append(ch_band_powers[band])
        
        # calculate average across channels
        for band in bands:
            if segment_band_powers[band]:
                avg_band_powers[band].append(np.mean(segment_band_powers[band]))
            else:
                avg_band_powers[band].append(0)
    
    # plot band powers over time
    band_colors = {
        'delta': '#FFD700',
        'theta': '#FF6347',
        'alpha': '#32CD32',
        'beta': '#1E90FF',
        'gamma': '#9370DB'
    }
    
    for band in bands:
        if avg_band_powers[band]:
            ax_band_time.plot(segment_times, avg_band_powers[band], 'o-', 
                             label=band.capitalize(), color=band_colors[band], linewidth=2)
    
    ax_band_time.set_title('average band power over time', fontsize=14)
    ax_band_time.set_xlabel('time (seconds)', fontsize=12)
    ax_band_time.set_ylabel('power', fontsize=12)
    ax_band_time.legend(loc='upper right')
    ax_band_time.grid(True)
    
    # 2. dominant frequency over time (top right)
    ax_freq_time = plt.subplot(gs[0, 1])
    
    # collect dominant frequencies across segments and channels
    dom_freqs = {ch: [] for ch in good_channels}
    
    for segment in segment_results:
        for ch in good_channels:
            if ch in segment['psd_results']:
                peak_freq = segment['psd_results'][ch]['peak_freq']
                dom_freqs[ch].append(peak_freq)
    
    # plot dominant frequency for each channel
    colors = plt.cm.viridis(np.linspace(0, 1, len(good_channels)))
    
    for i, ch in enumerate(good_channels):
        if all(len(segment_times) == len(dom_freqs[ch]) for segment in segment_results):
            ax_freq_time.plot(segment_times, dom_freqs[ch], 'o-', 
                             label=f'ch {ch}', color=colors[i], alpha=0.8)
    
    ax_freq_time.set_title('dominant frequency over time by channel', fontsize=14)
    ax_freq_time.set_xlabel('time (seconds)', fontsize=12)
    ax_freq_time.set_ylabel('frequency (hz)', fontsize=12)
    ax_freq_time.legend(loc='upper right', fontsize=8)
    ax_freq_time.grid(True)
    
    # 3. segment comparison: relative power profiles (middle left)
    ax_rel_power = plt.subplot(gs[1, 0])
    
    # calculate relative power for each segment
    segment_rel_powers = []
    
    for segment in segment_results:
        # initialize power sums
        band_sums = {band: 0 for band in bands}
        total_power = 0
        
        # sum powers across channels
        for ch in good_channels:
            if ch in segment['psd_results']:
                ch_band_powers = segment['psd_results'][ch]['band_powers']
                for band, power in ch_band_powers.items():
                    band_sums[band] += power
                    total_power += power
        
        # calculate relative powers
        rel_powers = {}
        if total_power > 0:
            for band in bands:
                rel_powers[band] = band_sums[band] / total_power * 100
        else:
            for band in bands:
                rel_powers[band] = 0
        
        segment_rel_powers.append(rel_powers)
    
    # create a grouped bar chart
    x = np.arange(len(bands))
    width = 0.8 / len(segment_rel_powers)
    
    for i, seg_powers in enumerate(segment_rel_powers):
        values = [seg_powers[band] for band in bands]
        offset = width * i - width * (len(segment_rel_powers) - 1) / 2
        ax_rel_power.bar(x + offset, values, width, 
                        label=f'segment {i+1}', alpha=0.7)
    
    ax_rel_power.set_title('relative band power by segment', fontsize=14)
    ax_rel_power.set_ylabel('relative power (%)', fontsize=12)
    ax_rel_power.set_xticks(x)
    ax_rel_power.set_xticklabels([band.capitalize() for band in bands])
    ax_rel_power.legend()
    ax_rel_power.grid(True, axis='y')
    
    # 4. alpha/beta ratio over time (middle right)
    ax_ratio = plt.subplot(gs[1, 1])
    
    # calculate alpha/beta ratio for each segment and channel
    alpha_beta_ratios = {ch: [] for ch in good_channels}
    
    for segment in segment_results:
        for ch in good_channels:
            if ch in segment['psd_results']:
                ch_band_powers = segment['psd_results'][ch]['band_powers']
                if 'alpha' in ch_band_powers and 'beta' in ch_band_powers:
                    alpha = ch_band_powers['alpha']
                    beta = ch_band_powers['beta']
                    if beta > 0:
                        ratio = alpha / beta
                        alpha_beta_ratios[ch].append(ratio)
                    else:
                        alpha_beta_ratios[ch].append(0)
    
    # plot alpha/beta ratios
    for i, ch in enumerate(good_channels):
        if alpha_beta_ratios[ch]:
            ax_ratio.plot(segment_times[:len(alpha_beta_ratios[ch])], alpha_beta_ratios[ch], 'o-', 
                         label=f'ch {ch}', color=colors[i], alpha=0.8)
    
    # add average line if there are multiple channels
    if len(good_channels) > 1:
        avg_ratios = []
        for i in range(len(segment_times)):
            segment_ratios = [alpha_beta_ratios[ch][i] for ch in good_channels 
                            if i < len(alpha_beta_ratios[ch])]
            if segment_ratios:
                avg_ratios.append(np.mean(segment_ratios))
            else:
                avg_ratios.append(0)
        
        ax_ratio.plot(segment_times[:len(avg_ratios)], avg_ratios, 'k-', 
                     label='average', linewidth=2)
    
    ax_ratio.set_title('alpha/beta power ratio over time', fontsize=14)
    ax_ratio.set_xlabel('time (seconds)', fontsize=12)
    ax_ratio.set_ylabel('alpha/beta ratio', fontsize=12)
    ax_ratio.legend(loc='upper right', fontsize=8)
    ax_ratio.grid(True)
    
    # 5. segment psd comparison for a representative channel (bottom)
    ax_psd_compare = plt.subplot(gs[2, :])
    
    # select the first good channel for comparison
    if good_channels:
        ch = good_channels[0]
        
        for i, segment in enumerate(segment_results):
            if ch in segment['psd_results']:
                freqs = segment['psd_results'][ch]['freqs']
                psd = segment['psd_results'][ch]['psd']
                
                # limit to 50 hz for better visualization
                mask = freqs <= 50
                ax_psd_compare.semilogy(freqs[mask], psd[mask], 
                                      label=f'segment {i+1} ({segment["start_time"]:.1f}s to {segment["end_time"]:.1f}s)',
                                      alpha=0.8)
        
        # add frequency band shading
        for band, (low, high) in FREQ_BANDS.items():
            if high <= 50:  # only show bands up to 50 hz
                ax_psd_compare.axvspan(low, high, color=band_colors[band], alpha=0.1, 
                                     label=f'{band.capitalize()} ({low} to {high} hz)')
        
        ax_psd_compare.set_title(f'psd comparison across segments for channel {ch}', fontsize=14)
        ax_psd_compare.set_xlabel('frequency (hz)', fontsize=12)
        ax_psd_compare.set_ylabel('power (log scale)', fontsize=12)
        ax_psd_compare.legend(loc='upper right', fontsize=8)
        ax_psd_compare.grid(True, which='both', linestyle='--', alpha=0.5)
        ax_psd_compare.set_xlim(0, 50)
    
    # add a title for the entire figure
    plt.suptitle('eeg segment comparison', fontsize=16, y=0.98)
    
    # adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # save the figure
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"segment comparison plot saved to {output_file}")
    plt.close()

def create_channel_correlation_plot(eeg_data, good_channels, output_file):
    """
    create a plot showing channel correlations and connectivity.
    
    args:
        eeg_data: eeg data [channels, frames]
        good_channels: list of good channel indices
        output_file: output file path
    """
    if not good_channels or len(good_channels) < 2:
        print("not enough good channels for correlation analysis")
        return
    
    # create figure
    plt.figure(figsize=(18, 10))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # set up grid
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])
    
    # 1. correlation matrix (left)
    ax_corr = plt.subplot(gs[0, 0])
    
    # calculate correlation matrix for good channels
    good_data = eeg_data[good_channels, :]
    corr_matrix = np.corrcoef(good_data)
    
    # create a custom colormap: blue for negative, white for zero, red for positive
    cmap = LinearSegmentedColormap.from_list(
        'blue_white_red',
        [(0, 'blue'), (0.5, 'white'), (1.0, 'red')],
        N=256
    )
    
    # plot correlation matrix
    im = ax_corr.imshow(corr_matrix, cmap=cmap, vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax_corr, label='correlation coefficient')
    
    # add channel labels
    ax_corr.set_xticks(np.arange(len(good_channels)))
    ax_corr.set_yticks(np.arange(len(good_channels)))
    ax_corr.set_xticklabels(good_channels)
    ax_corr.set_yticklabels(good_channels)
    plt.setp(ax_corr.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # add correlation values as text
    for i in range(len(good_channels)):
        for j in range(len(good_channels)):
            text_color = "white" if abs(corr_matrix[i, j]) > 0.5 else "black"
            ax_corr.text(j, i, f"{corr_matrix[i, j]:.2f}", 
                        ha="center", va="center", color=text_color, fontsize=8)
    
    ax_corr.set_title('channel correlation matrix', fontsize=14)
    
    # 2. high correlation network (right)
    ax_network = plt.subplot(gs[0, 1])
    
    # define positions for each channel (circle layout)
    angles = np.linspace(0, 2*np.pi, len(good_channels), endpoint=False)
    pos_x = np.cos(angles)
    pos_y = np.sin(angles)
    
    # plot channel nodes
    ax_network.scatter(pos_x, pos_y, s=200, c='skyblue', edgecolor='blue', zorder=3)
    
    # add channel labels
    for i, ch in enumerate(good_channels):
        ax_network.text(pos_x[i]*1.1, pos_y[i]*1.1, f"{ch}", 
                       ha='center', va='center', fontsize=10, fontweight='bold')
    
    # draw edges for correlations above threshold
    threshold = 0.7  # only show strong correlations
    
    for i in range(len(good_channels)):
        for j in range(i+1, len(good_channels)):
            corr = corr_matrix[i, j]
            if abs(corr) > threshold:
                # line width proportional to correlation strength
                lw = abs(corr) * 3
                
                # line color based on positive/negative correlation
                color = 'red' if corr > 0 else 'blue'
                
                # alpha also proportional to strength
                alpha = 0.2 + 0.8 * abs(corr)
                
                # draw the connection line
                ax_network.plot([pos_x[i], pos_x[j]], [pos_y[i], pos_y[j]], 
                              color=color, linewidth=lw, alpha=alpha, zorder=1)
                
                # add correlation value at the middle of the line
                mid_x = (pos_x[i] + pos_x[j]) / 2
                mid_y = (pos_y[i] + pos_y[j]) / 2
                ax_network.text(mid_x, mid_y, f"{corr:.2f}", 
                              ha='center', va='center', fontsize=8,
                              bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    ax_network.set_title(f'channel correlation network (threshold: {threshold})', fontsize=14)
    ax_network.set_xlim(-1.5, 1.5)
    ax_network.set_ylim(-1.5, 1.5)
    ax_network.set_aspect('equal')
    ax_network.axis('off')
    
    # add a title for the entire figure
    plt.suptitle('channel correlation analysis', fontsize=16, y=0.98)
    
    # adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # save the figure
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"channel correlation plot saved to {output_file}")
    plt.close()

def create_channel_detail_plot(eeg_data, sample_rate, channel, results, output_file):
    """
    create a detailed plot for a specific channel.
    
    args:
        eeg_data: eeg data [channels, frames]
        sample_rate: sampling rate in hz
        channel: channel index to analyze
        results: analysis results
        output_file: output file path
    """
    # extract channel data
    ch_data = eeg_data[channel, :]
    
    # check if we have results for this channel
    has_psd_results = channel in results['psd_results']
    has_fooof_results = channel in results['fooof_results'] and results['fooof_results'][channel] is not None
    
    # create figure
    plt.figure(figsize=(18, 15))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # set up grid
    gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1])
    
    # 1. time series (top left)
    ax_time = plt.subplot(gs[0, 0])
    
    # show first 10 seconds
    seconds_to_show = 10
    samples_to_show = min(int(seconds_to_show * sample_rate), ch_data.shape[0])
    
    ax_time.plot(np.arange(samples_to_show) / sample_rate, ch_data[:samples_to_show], 'b-')
    
    ax_time.set_title(f'channel {channel} time series (first 10 seconds)', fontsize=14)
    ax_time.set_xlabel('time (seconds)', fontsize=12)
    ax_time.set_ylabel('amplitude', fontsize=12)
    ax_time.grid(True)
    
    # 2. signal statistics (top right)
    ax_stats = plt.subplot(gs[0, 1])
    
    # calculate statistics
    mean = np.mean(ch_data)
    median = np.median(ch_data)
    std_dev = np.std(ch_data)
    min_val = np.min(ch_data)
    max_val = np.max(ch_data)
    range_val = max_val - min_val
    rms = np.sqrt(np.mean(np.square(ch_data)))
    
    # calculate histogram
    hist, bins = np.histogram(ch_data, bins=50, density=True)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # plot histogram
    ax_stats.bar(bin_centers, hist, width=(bins[1] - bins[0]), alpha=0.7, color='blue')
    
    # add normal distribution with same mean/std
    x = np.linspace(min_val, max_val, 100)
    pdf = 1 / (std_dev * np.sqrt(2 * np.pi)) * np.exp(-(x - mean) ** 2 / (2 * std_dev ** 2))
    ax_stats.plot(x, pdf, 'r-', linewidth=2, label='normal distribution')
    
    # add statistics as text
    stats_text = f"mean: {mean:.4f}\nmedian: {median:.4f}\nstd dev: {std_dev:.4f}"
    stats_text += f"\nmin: {min_val:.4f}\nmax: {max_val:.4f}\nrange: {range_val:.4f}"
    stats_text += f"\nrms: {rms:.4f}"
    
    ax_stats.text(0.02, 0.98, stats_text, transform=ax_stats.transAxes, 
                 fontsize=10, va='top', 
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax_stats.set_title(f'channel {channel} amplitude distribution', fontsize=14)
    ax_stats.set_xlabel('amplitude', fontsize=12)
    ax_stats.set_ylabel('probability density', fontsize=12)
    ax_stats.legend()
    ax_stats.grid(True)
    
    # 3. power spectral density (middle left)
    ax_psd = plt.subplot(gs[1, 0])
    
    if has_psd_results:
        # get psd from results
        freqs = results['psd_results'][channel]['freqs']
        psd = results['psd_results'][channel]['psd']
        
        # limit to 50 hz for better visualization
        mask = freqs <= 50
        ax_psd.semilogy(freqs[mask], psd[mask], 'b-', label='power spectral density')
        
        # add frequency band shading
        band_colors = {
            'delta': '#FFD700',
            'theta': '#FF6347',
            'alpha': '#32CD32',
            'beta': '#1E90FF',
            'gamma': '#9370DB'
        }
        
        for band, (low, high) in FREQ_BANDS.items():
            if high <= 50:  # only show bands up to 50 hz
                ax_psd.axvspan(low, high, color=band_colors[band], alpha=0.2, 
                             label=f'{band.capitalize()} ({low} to {high} hz)')
        
        # add band power values as text
        band_powers = results['psd_results'][channel]['band_powers']
        band_text = "band powers:\n"
        for band, power in band_powers.items():
            band_text += f"{band.capitalize()}: {power:.6f}\n"
        
        ax_psd.text(0.02, 0.98, band_text, transform=ax_psd.transAxes, 
                   fontsize=10, va='top', 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        # calculate psd on the fly
        nperseg = min(4096, len(ch_data))
        nfft = max(nperseg, 2**int(np.ceil(np.log2(nperseg))))
        freqs, psd = signal.welch(ch_data, fs=sample_rate, nperseg=nperseg, nfft=nfft)
        
        # limit to 50 hz for better visualization
        mask = freqs <= 50
        ax_psd.semilogy(freqs[mask], psd[mask], 'b-', label='power spectral density')
    
    ax_psd.set_title(f'channel {channel} power spectral density', fontsize=14)
    ax_psd.set_xlabel('frequency (hz)', fontsize=12)
    ax_psd.set_ylabel('power (log scale)', fontsize=12)
    ax_psd.legend(loc='upper right', fontsize=8)
    ax_psd.grid(True, which='both', linestyle='--', alpha=0.5)
    ax_psd.set_xlim(0, 50)
    
    # 4. fooof analysis (middle right)
    ax_fooof = plt.subplot(gs[1, 1])
    
    if has_fooof_results:
        # get fooof results
        fooof_data = results['fooof_results'][channel]
        
        # plot original spectrum
        freqs = fooof_data['freqs']
        power_spectrum = fooof_data['power_spectrum']
        ax_fooof.semilogy(freqs, power_spectrum, 'k-', label='original spectrum', alpha=0.8)
        
        # plot fooof model fit
        fooofed_spectrum = fooof_data['fooofed_spectrum']
        ax_fooof.semilogy(freqs, fooofed_spectrum, 'r-', label='fooof model fit', alpha=0.8)
        
        # plot aperiodic component
        aperiodic_fit = fooof_data['aperiodic_fit']
        ax_fooof.semilogy(freqs, aperiodic_fit, 'b--', label='aperiodic component', alpha=0.8)
        
        # plot periodic component (subtract aperiodic from original)
        periodic_component = fooof_data['periodic_component']
        
        # add a second y-axis for the periodic component
        ax_periodic = ax_fooof.twinx()
        ax_periodic.plot(freqs, periodic_component, 'g-', label='periodic component', alpha=0.6)
        ax_periodic.set_ylabel('periodic power', color='green', fontsize=12)
        ax_periodic.tick_params(axis='y', labelcolor='green')
        
        # add detected peaks as vertical spans
        peak_params = fooof_data['peak_params']
        for peak in peak_params:
            cf, pw, bw = peak
            # calculate gaussian width (2*std)
            width = bw / 2
            ax_fooof.axvspan(cf - width, cf + width, color='green', alpha=0.2)
            ax_fooof.axvline(cf, color='green', linestyle='-', alpha=0.5, linewidth=1)
        
        # add model fit statistics
        r_squared = fooof_data['r_squared']
        error = fooof_data['error']
        aperiodic_params = fooof_data['aperiodic_params']
        
        # add aperiodic parameters text
        ax_fooof.text(0.02, 0.98, 
                    f"aperiodic:\noffset: {aperiodic_params[0]:.2f}\nexponent: {aperiodic_params[1]:.2f}\nr²: {r_squared:.4f}",
                    transform=ax_fooof.transAxes, fontsize=10, va='top', 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # add band peaks if available
        if 'peaks_by_band' in fooof_data:
            peak_text = "oscillatory peaks:\n"
            for band, peak in fooof_data['peaks_by_band'].items():
                if peak is not None:
                    peak_text += f"{band.capitalize()}: {peak['cf']:.2f} hz, "
                    peak_text += f"power: {peak['power']:.4f}, "
                    peak_text += f"bw: {peak['bw']:.2f}\n"
                else:
                    peak_text += f"{band.capitalize()}: none\n"
            
            ax_fooof.text(0.65, 0.98, peak_text, transform=ax_fooof.transAxes, 
                        fontsize=9, va='top', 
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        ax_fooof.text(0.5, 0.5, "no fooof analysis available for this channel", 
                    transform=ax_fooof.transAxes, fontsize=12, ha='center', va='center')
    
    ax_fooof.set_title(f'channel {channel} fooof analysis', fontsize=14)
    ax_fooof.set_xlabel('frequency (hz)', fontsize=12)
    ax_fooof.set_ylabel('power (log scale)', fontsize=12)
    ax_fooof.legend(loc='upper right', fontsize=8)
    ax_fooof.grid(True, which='both', linestyle='--', alpha=0.5)
    ax_fooof.set_xlim(0, 45)
    
    # 5. spectrogram (bottom)
    ax_specgram = plt.subplot(gs[2, :])
    
    # calculate spectrogram
    nperseg = min(int(2 * sample_rate), len(ch_data))  # 2-second window
    noverlap = nperseg // 2  # 50% overlap
    
    f, t, Sxx = signal.spectrogram(ch_data, fs=sample_rate, nperseg=nperseg, noverlap=noverlap)
    
    # plot spectrogram up to 50 hz
    f_mask = f <= 50
    im = ax_specgram.pcolormesh(t, f[f_mask], 10 * np.log10(Sxx[f_mask, :]), 
                              shading='gouraud', cmap='viridis')
    
    # add colorbar
    cbar = plt.colorbar(im, ax=ax_specgram)
    cbar.set_label('power/frequency (db/hz)', fontsize=10)
    
    # add frequency band lines
    band_colors = {
        'delta': '#FFD700',
        'theta': '#FF6347',
        'alpha': '#32CD32',
        'beta': '#1E90FF',
        'gamma': '#9370DB'
    }
    
    for band, (low, high) in FREQ_BANDS.items():
        if high <= 50:  # only show bands up to 50 hz
            ax_specgram.axhline(y=low, color=band_colors[band], linestyle='--', alpha=0.8, linewidth=1)
            ax_specgram.axhline(y=high, color=band_colors[band], linestyle='--', alpha=0.8, linewidth=1)
            # add band label
            ax_specgram.text(t[-1]*1.01, (low+high)/2, band.capitalize(), 
                           color=band_colors[band], va='center', ha='left', fontsize=8)
    
    ax_specgram.set_title(f'channel {channel} spectrogram', fontsize=14)
    ax_specgram.set_xlabel('time (seconds)', fontsize=12)
    ax_specgram.set_ylabel('frequency (hz)', fontsize=12)
    
    # add a title for the entire figure
    plt.suptitle(f'channel {channel} detailed analysis', fontsize=16, y=0.98)
    
    # adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # save the figure
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"channel {channel} detail plot saved to {output_file}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='enhanced eeg analysis for gtec data')
    parser.add_argument('input_file', nargs='?', type=str, help='specific h5 file to analyze')
    parser.add_argument('--file', dest='file_arg', type=str, help='specific h5 file to analyze')
    parser.add_argument('--dir', type=str, default='./data',
                        help='directory containing h5 files (default: ./data)')
    parser.add_argument('--duration', type=int, default=None,
                        help='duration in seconds to analyze (default: full recording)')
    parser.add_argument('--list-files', action='store_true',
                        help='just list available h5 files without analysis')
    parser.add_argument('--exclude-channels', type=str,
                        help='comma-separated list of channel indices to exclude from analysis (e.g., "8,10,12")')
    parser.add_argument('--low-freq', type=float, default=0.5,
                        help='lower cutoff frequency for bandpass filter in hz (default: 0.5)')
    parser.add_argument('--high-freq', type=float, default=45.0,
                        help='upper cutoff frequency for bandpass filter in hz (default: 45.0)')
    parser.add_argument('--num-segments', type=int, default=5,
                        help='number of time segments to divide the recording into (default: 5)')
    parser.add_argument('--full-file', action='store_true',
                        help='analyze the entire file rather than a random segment')
    args = parser.parse_args()

    # suppress warnings to avoid cluttering the console
    warnings.filterwarnings("ignore", category=UserWarning)
    
    print("\n=== enhanced eeg analysis tool ===")
    print("this script provides detailed analysis of eeg data with time segmentation")
    print("and separation of periodic/aperiodic components using fooof.\n")

    # determine the file to analyze (prioritize positional argument, then --file flag)
    file_to_analyze = None
    if args.input_file:
        file_to_analyze = args.input_file
        print(f"using file from positional argument: {file_to_analyze}")
    elif args.file_arg:
        file_to_analyze = args.file_arg
        print(f"using file from --file argument: {file_to_analyze}")

    # find all h5 files in the specified directory
    h5_files = []
    for root, dirs, files in os.walk(args.dir):
        for file in files:
            if file.endswith(".h5"):
                h5_files.append(os.path.join(root, file))
    
    print(f"found {len(h5_files)} h5 files in {args.dir}")

    if args.list_files:
        for i, file in enumerate(h5_files):
            print(f"{i+1}. {file}")
        return

    # parse excluded channels if provided
    exclude_channels = None
    if args.exclude_channels:
        try:
            exclude_channels = [int(ch.strip()) for ch in args.exclude_channels.split(',')]
            print(f"will exclude channels {exclude_channels} from analysis")
        except ValueError:
            print(f"warning: invalid channel format in --exclude-channels. expected comma-separated integers.")

    # analyze the appropriate file
    if file_to_analyze:
        print(f"attempting to analyze file: {file_to_analyze}")
        # first check if the file exists as specified
        if os.path.isfile(file_to_analyze):
            print(f"file exists, analyzing: {file_to_analyze}")
            analyze_gtec_eeg_data(file_to_analyze, args.duration, exclude_channels,
                                args.low_freq, args.high_freq, args.num_segments,
                                args.full_file)
        else:
            # if doesn't exist, try to find it in the directory
            print(f"file not found directly. searching in {args.dir}...")
            basename = os.path.basename(file_to_analyze)
            matching_files = [f for f in h5_files if basename in f]
            if matching_files:
                print(f"found matching file: {matching_files[0]}")
                analyze_gtec_eeg_data(matching_files[0], args.duration, exclude_channels,
                                    args.low_freq, args.high_freq, args.num_segments,
                                    args.full_file)
            else:
                print(f"error: file not found: {file_to_analyze}")
    else:
        # no file specified, use the first file from the directory
        if h5_files:
            print(f"no file specified. analyzing first h5 file: {h5_files[0]}")
            analyze_gtec_eeg_data(h5_files[0], args.duration, exclude_channels,
                                args.low_freq, args.high_freq, args.num_segments,
                                args.full_file)
        else:
            print("no h5 files found to analyze")

if __name__ == "__main__":
    main()


