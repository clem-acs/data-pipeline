"""
Analysis transform for EEG data in the pipeline.

This transform:
1. Reads curated H5 files from S3 (curated-h5/)
2. Loads the EEG data (first 20 channels)
3. Calculates 60Hz noise power (59-61Hz band) as a fraction of total power for each channel
4. Generates visualizations of 60Hz noise power ratios across channels
5. Saves analysis reports as JSON to processed/analysis/ prefix in S3
6. Records metadata in DynamoDB

This is implemented using the BaseTransform architecture.
"""

import os
import sys
import json
import h5py
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional, Set

# Import base transform
from base_transform import BaseTransform, Session


class AnalysisTransform(BaseTransform):
    """
    Analysis transform for EEG data.

    This transform analyzes EEG data and produces summary reports:
    1. Computes 60Hz noise power as a fraction of total power for each EEG channel
    2. Identifies channels with excessive line noise (above common thresholds)
    3. Generates visualizations of 60Hz noise metrics
    4. Outputs a JSON analysis report with detailed power statistics

    Note: EEG data is assumed to be in microvolts (µV) as per standard practice.
    """

    # Define required class attributes for source and destination
    SOURCE_PREFIX = 'curated-h5/'
    DEST_PREFIX = 'processed/analysis/'

    def __init__(self,
                 plot_60hz_noise: bool = True,
                 **kwargs):
        """
        Initialize the analysis transform.

        Args:
            plot_60hz_noise: Whether to generate plots for the 60Hz noise analysis
            **kwargs: Additional arguments for BaseTransform
        """
        # Set default transform info if not provided
        transform_id = kwargs.pop('transform_id', 'analysis_v0')
        script_id = kwargs.pop('script_id', '2E')
        script_name = kwargs.pop('script_name', 'analyze_eeg')
        script_version = kwargs.pop('script_version', 'v0')

        # Call parent constructor
        super().__init__(
            transform_id=transform_id,
            script_id=script_id,
            script_name=script_name,
            script_version=script_version,
            **kwargs
        )

        # Set analysis-specific attributes
        self.plot_iqr = plot_60hz_noise  # Keep variable name for compatibility

        self.logger.info(f"EEG Analysis transform initialized with:")
        self.logger.info(f"  Plot 60Hz Noise: {self.plot_iqr}")
        if self.keep_local:
            self.logger.info(f"  KEEP LOCAL: Will keep temporary files for inspection")

    def process_session(self, session: Session) -> Dict:
        """Process a single session.

        This implementation:
        1. Finds the curated H5 file for the session
        2. Loads the EEG data
        3. Computes inter-quartile range for each channel
        4. Creates an analysis report

        Args:
            session: Session object

        Returns:
            Dict with processing results
        """
        session_id = session.session_id

        self.logger.info(f"Processing session: {session_id}")

        # In curated-h5/, files are always directly in the source prefix
        curated_h5_key = f"{self.source_prefix}{session_id}.h5"

        # Check if the file exists
        try:
            self.s3.head_object(Bucket=self.s3_bucket, Key=curated_h5_key)
            self.logger.info(f"Found curated H5 file: {curated_h5_key}")
        except Exception as e:
            self.logger.error(f"No curated H5 file found for session {session_id}: {e}")

            return {
                "status": "failed",
                "error_details": f"No curated H5 file found for session {session_id}",
                "metadata": {"session_id": session_id},
                "files_to_copy": [],
                "files_to_upload": []
            }

        # Download the H5 file
        local_h5_path = session.download_file(curated_h5_key)

        try:
            # Analyze the EEG data
            analysis_results = self.analyze_eeg_data(local_h5_path, session_id)

            # Generate output files (reports and plots)
            output_files = self.generate_output_files(analysis_results, session)

            # Create metadata from analysis results
            # Calculate avg 60Hz noise ratio for summary
            noise_ratios = analysis_results.get("noise_60hz_ratio", [])
            avg_noise_ratio = 0.0
            if noise_ratios:
                avg_noise_ratio = sum(noise_ratios) / len(noise_ratios)
            
            metadata = {
                "session_id": session_id,
                "has_eeg": analysis_results.get("has_eeg", False),
                "total_eeg_channels": analysis_results.get("total_eeg_channels", 0),
                "analyzed_eeg_channels": analysis_results.get("analyzed_eeg_channels", 0),
                "eeg_samples": analysis_results.get("eeg_sample_count", 0),
                "avg_60hz_noise_ratio": float(avg_noise_ratio),
                "analysis_type": "eeg_60hz_noise"
            }

            # Prepare files to upload
            files_to_upload = []
            for local_path, dest_key in output_files:
                files_to_upload.append((local_path, dest_key))

            return {
                "status": "success",
                "metadata": metadata,
                "files_to_copy": [],
                "files_to_upload": files_to_upload
            }

        except Exception as e:
            self.logger.error(f"Error analyzing session {session_id}: {e}", exc_info=True)

            return {
                "status": "failed",
                "error_details": str(e),
                "metadata": {"session_id": session_id},
                "files_to_copy": [],
                "files_to_upload": []
            }

    def analyze_eeg_data(self, file_path: str, session_id: str) -> Dict:
        """Analyze the EEG data in a curated H5 file to calculate 60Hz noise.

        Args:
            file_path: Path to the local H5 file
            session_id: Session ID

        Returns:
            Dict with analysis results
        """
        self.logger.info(f"Analyzing EEG data for session {session_id}")

        # Initialize results
        analysis_results = {
            "session_id": session_id,
            "has_eeg": False,
            "eeg_channel_count": 0,
            "eeg_sample_count": 0,
            "noise_60hz_ratio": [],
            "timestamps": []
        }

        try:
            with h5py.File(file_path, 'r') as f:
                # Check if the file has EEG data from standard locations
                if 'devices/eeg/timestamps' in f and 'devices/eeg/frames_data' in f:
                    analysis_results["has_eeg"] = True
                    self.logger.info(f"Found EEG data in devices/eeg")

                    # Load EEG data and sampling rate
                    frames_data = f['devices/eeg/frames_data'][()]
                    timestamps = f['devices/eeg/timestamps'][()]
                    
                    # Calculate sampling rate from timestamps
                    try:
                        if len(timestamps) > 1:
                            time_diff = np.mean(np.diff(timestamps))
                            # Check if time_diff is valid
                            if time_diff > 0:
                                sampling_rate = 1.0 / time_diff
                            else:
                                # Default if timestamps are invalid
                                sampling_rate = 500.0
                                self.logger.warning(f"Invalid timestamp differences detected, using default sampling rate")
                        else:
                            # Default to common EEG sampling rate if can't determine
                            sampling_rate = 500.0
                            self.logger.warning(f"Not enough timestamps to calculate sampling rate, using default")
                    except Exception as e:
                        # Fallback to default sampling rate
                        sampling_rate = 500.0
                        self.logger.warning(f"Error calculating sampling rate: {e}, using default 500 Hz")
                    
                    self.logger.info(f"EEG frames_data shape: {frames_data.shape}, Sampling rate: {sampling_rate:.2f} Hz")

                    # Process the data - data format is [chunks, channels, samples]
                    chunks = frames_data.shape[0]
                    n_channels = frames_data.shape[1]
                    samples_per_chunk = frames_data.shape[2]

                    # Reshape to get [channels, total_samples]
                    rearranged = np.transpose(frames_data, (1, 0, 2))
                    eeg_data = rearranged.reshape(n_channels, -1)
                    
                    # Get the configured channel count from channels_json attribute if available
                    n_configured_channels = None
                    try:
                        if 'channels_json' in f['devices/eeg'].attrs:
                            channels_json = f['devices/eeg'].attrs['channels_json']
                            if isinstance(channels_json, str):
                                # Parse the JSON string to get the list of channel names
                                import json
                                channel_names = json.loads(channels_json)
                                n_configured_channels = len(channel_names)
                                self.logger.info(f"Found {n_configured_channels} configured EEG channels from channels_json")
                    except Exception as e:
                        self.logger.error(f"ERROR READING CHANNELS_JSON ATTRIBUTE: {e}")
                        self.logger.error(f"This could affect analysis accuracy")
                    
                    # Use configured channel count if available, otherwise default to 21 channels
                    if n_configured_channels is not None:
                        channels_to_analyze = min(n_configured_channels, n_channels)
                        self.logger.info(f"Analyzing first {channels_to_analyze} configured EEG channels")
                    else:
                        # Default to 21 channels if configuration is not available
                        channels_to_analyze = min(21, n_channels)
                        self.logger.error(f"CRITICAL: No channel configuration found, defaulting to {channels_to_analyze} EEG channels")
                        self.logger.error(f"This could affect analysis accuracy if the actual configuration differs")
                    
                    eeg_data = eeg_data[:channels_to_analyze, :]

                    analysis_results["total_eeg_channels"] = n_channels
                    analysis_results["analyzed_eeg_channels"] = channels_to_analyze
                    analysis_results["eeg_sample_count"] = chunks * samples_per_chunk
                    analysis_results["n_configured_channels"] = n_configured_channels
                    analysis_results["sampling_rate"] = float(sampling_rate)

                    # Calculate 60Hz noise ratio for each channel using 10s windows
                    from scipy import signal
                    
                    # Calculate window parameters
                    window_seconds = 10
                    samples_per_window = int(window_seconds * sampling_rate)
                    total_samples = eeg_data.shape[1]
                    
                    # Ensure we have valid window parameters
                    if samples_per_window <= 0:
                        self.logger.warning(f"Invalid window size: {samples_per_window} samples. Adjusting to 1/5 of total samples.")
                        samples_per_window = max(1000, total_samples // 5)
                        window_seconds = samples_per_window / max(1.0, sampling_rate)
                    
                    # Create windows (with 50% overlap for better averaging)
                    step_size = max(1, samples_per_window // 2)  # Ensure step size is at least 1
                    
                    # Calculate number of windows, ensuring no division by zero
                    if step_size > 0:
                        n_windows = max(1, (total_samples - samples_per_window) // step_size + 1)
                    else:
                        n_windows = 1  # Fallback to a single window
                    
                    self.logger.info(f"Analyzing {n_windows} windows of {window_seconds}s each (50% overlap)")
                    
                    # Parameters for binned FFT
                    bin_width_hz = 2  # 2 Hz bins
                    max_freq_hz = min(200, sampling_rate / 2)  # Nyquist limit or 200 Hz, whichever is smaller
                    
                    analysis_results["noise_60hz_ratio"] = []
                    analysis_results["noise_60hz_by_channel"] = {}
                    analysis_results["window_seconds"] = window_seconds
                    analysis_results["window_count"] = n_windows
                    analysis_results["fft_bin_width_hz"] = bin_width_hz
                    analysis_results["fft_max_freq_hz"] = float(max_freq_hz)
                    
                    # Create bins for the binned FFT
                    freq_bins = np.arange(0, max_freq_hz + bin_width_hz, bin_width_hz)
                    bin_centers = freq_bins[:-1] + bin_width_hz / 2
                    
                    # Initialize binned FFT average for all channels
                    all_channels_binned_fft = np.zeros(len(bin_centers))
                    binned_fft_by_channel = {}

                    for channel in range(channels_to_analyze):
                        channel_data = eeg_data[channel, :]
                        
                        # Initialize arrays to store window results
                        window_power_ratios = []
                        window_60hz_powers = []
                        window_total_powers = []
                        
                        # Initialize binned FFT for this channel
                        channel_binned_fft = np.zeros(len(bin_centers))
                        window_count = 0
                        
                        # Process each window
                        for i in range(n_windows):
                            start_idx = i * step_size
                            end_idx = min(start_idx + samples_per_window, total_samples)
                            
                            # Skip windows that are too short
                            if end_idx - start_idx < samples_per_window / 2:
                                continue
                                
                            window_data = channel_data[start_idx:end_idx]
                            
                            # Apply Hanning window to reduce spectral leakage
                            window_data = window_data * np.hanning(len(window_data))
                            
                            # Calculate power spectrum with appropriate nperseg
                            # Use a segment length that gives ~0.5 Hz frequency resolution
                            # Ensure nperseg is valid (must be >= 2)
                            nperseg = min(int(max(2, 2 * sampling_rate)), len(window_data))
                            nperseg = max(nperseg, 2)  # Ensure minimum valid value
                            
                            try:
                                freqs, psd = signal.welch(window_data, fs=max(1.0, sampling_rate), nperseg=nperseg)
                            except Exception as e:
                                self.logger.warning(f"Error in welch calculation: {e}. Skipping window.")
                                continue
                            
                            # Find indices for 60Hz +/- 1Hz band
                            idx_60hz = np.where((freqs >= 59) & (freqs <= 61))[0]
                            
                            if len(idx_60hz) > 0:  # Ensure we found the 60Hz band
                                # Calculate power ratio for this window
                                power_60hz = np.sum(psd[idx_60hz])
                                total_power = np.sum(psd)
                                
                                if total_power > 0:  # Avoid division by zero
                                    power_ratio = float(power_60hz / total_power)
                                    
                                    # Store this window's results
                                    window_power_ratios.append(power_ratio)
                                    window_60hz_powers.append(float(power_60hz))
                                    window_total_powers.append(float(total_power))
                            
                            # Calculate binned FFT - bin PSD values into 2 Hz bins
                            for bin_idx in range(len(bin_centers)):
                                bin_start = freq_bins[bin_idx]
                                bin_end = freq_bins[bin_idx + 1]
                                
                                # Find indices that fall within this bin
                                bin_indices = np.where((freqs >= bin_start) & (freqs < bin_end))[0]
                                
                                if len(bin_indices) > 0:
                                    # Add the average power in this bin
                                    bin_power = np.mean(psd[bin_indices])
                                    channel_binned_fft[bin_idx] += bin_power
                            
                            window_count += 1
                        
                        # Calculate average metrics across all windows
                        avg_power_ratio = 0.0
                        if window_power_ratios:
                            avg_power_ratio = float(np.mean(window_power_ratios))
                            
                        # Store the channel's average metrics
                        analysis_results["noise_60hz_ratio"].append(avg_power_ratio)
                        
                        # Calculate average binned FFT for this channel and add to all-channel average
                        if window_count > 0:
                            # Normalize by window count to get average
                            channel_binned_fft = channel_binned_fft / window_count
                            
                            # Add to the all-channels average
                            all_channels_binned_fft += channel_binned_fft
                            
                            # Store this channel's binned FFT
                            binned_fft_by_channel[f"channel_{channel}"] = channel_binned_fft.tolist()
                        
                        # Store detailed channel stats
                        analysis_results["noise_60hz_by_channel"][f"channel_{channel}"] = {
                            "power_ratio_avg": avg_power_ratio,
                            "power_ratio_max": float(np.max(window_power_ratios)) if window_power_ratios else 0.0,
                            "power_ratio_min": float(np.min(window_power_ratios)) if window_power_ratios else 0.0,
                            "power_ratio_std": float(np.std(window_power_ratios)) if window_power_ratios else 0.0,
                            "window_count": len(window_power_ratios),
                            "avg_power_60hz": float(np.mean(window_60hz_powers)) if window_60hz_powers else 0.0,
                            "avg_total_power": float(np.mean(window_total_powers)) if window_total_powers else 0.0
                        }
                else:
                    self.logger.warning(f"No EEG device found in the file")

                # Calculate the average FFT across all channels
                if channels_to_analyze > 0:
                    # Normalize by number of channels
                    all_channels_binned_fft = all_channels_binned_fft / channels_to_analyze
                    
                    # Store the binned FFT data in the results
                    analysis_results["fft_bin_centers"] = bin_centers.tolist()
                    analysis_results["avg_binned_fft"] = all_channels_binned_fft.tolist()
                    analysis_results["binned_fft_by_channel"] = binned_fft_by_channel
                
                self.logger.info(f"Analysis complete: 60Hz noise ratio and binned FFT computed for {analysis_results.get('analyzed_eeg_channels', 0)} EEG channels")

        except Exception as e:
            self.logger.error(f"Error analyzing 60Hz noise for {session_id}: {e}", exc_info=True)
            raise

        return analysis_results

    def generate_output_files(self, analysis_results: Dict, session: Session) -> List[tuple]:
        """Generate output files from analysis results.

        Args:
            analysis_results: Analysis results dictionary
            session: Session object

        Returns:
            List of tuples (local_path, destination_key)
        """
        session_id = session.session_id
        output_files = []

        # Save the analysis report as JSON
        report_file_name = f"{session_id}_eeg_60hz_noise_analysis.json"
        local_report_path = session.create_upload_file(report_file_name)

        with open(local_report_path, 'w') as f:
            json.dump(analysis_results, f, indent=2)

        dest_key = f"{self.destination_prefix}{report_file_name}"
        output_files.append((local_report_path, dest_key))

        # Generate plots if enabled
        if self.plot_iqr and analysis_results.get("has_eeg", False):
            # Generate channel metrics plot
            noise_ratios = analysis_results.get("noise_60hz_ratio", [])
            if noise_ratios and len(noise_ratios) > 0:
                # Create metrics plot
                plot_file_name = f"{session_id}_eeg_60hz_noise.png"
                local_plot_path = session.create_upload_file(plot_file_name)

                # Get data for plotting
                channels = [f"Ch {i}" for i in range(len(noise_ratios))]

                # Create figure
                fig, ax = plt.subplots(figsize=(14, 8))

                # Plot 60Hz noise ratio with gradient color based on values
                # Define a colormap that goes from green (low) to yellow to red (high)
                cmap = plt.cm.get_cmap('RdYlGn_r')
                
                # Get max value for color normalization (use at least 0.3 for scale)
                max_val = max(max(noise_ratios), 0.3)
                colors = [cmap(val/max_val) for val in noise_ratios]
                
                bars = ax.bar(channels, noise_ratios, color=colors)
                
                # Add a more informative title with window info
                window_count = analysis_results.get("window_count", 0)
                window_seconds = analysis_results.get("window_seconds", 10)
                ax.set_title(f"EEG 60Hz Noise Power Ratio - {window_count} windows of {window_seconds}s each")
                
                ax.set_xlabel("Channel")
                ax.set_ylabel("60Hz Noise Ratio (Power in 59-61Hz / Total Power)")
                ax.grid(axis='y', linestyle='--', alpha=0.7)

                # Add value labels to bars
                for bar in bars:
                    height = bar.get_height()
                    # Only show label if value is significant (reduces clutter for small values)
                    if height > 0.001:
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                                f'{height:.4f}', ha='center', va='bottom', fontsize=8, 
                                color='black' if height < 0.1 else 'white')

                # Add horizontal line at common thresholds with better labels
                ax.axhline(y=0.05, color='yellow', linestyle='--', alpha=0.7, label='5% - Moderate noise')
                ax.axhline(y=0.1, color='orange', linestyle='--', alpha=0.7, label='10% - High noise')
                ax.axhline(y=0.2, color='red', linestyle='--', alpha=0.7, label='20% - Severe noise')
                ax.legend(loc='upper right')

                # Add rotation to x-labels if many channels
                if len(channels) > 10:
                    plt.setp(ax.get_xticklabels(), rotation=90)

                plt.tight_layout()
                plt.savefig(local_plot_path)
                plt.close()

                # Add to output files
                dest_key = f"{self.destination_prefix}{plot_file_name}"
                output_files.append((local_plot_path, dest_key))
                
            # Create full FFT plot if data is available
            if "fft_bin_centers" in analysis_results and "avg_binned_fft" in analysis_results:
                # Create FFT plot
                fft_plot_file_name = f"{session_id}_eeg_binned_fft.png"
                local_fft_path = session.create_upload_file(fft_plot_file_name)
                
                # Get data for plotting
                bin_centers = analysis_results.get("fft_bin_centers", [])
                avg_fft = analysis_results.get("avg_binned_fft", [])
                
                if bin_centers and avg_fft:
                    # Create figure
                    fig, ax = plt.subplots(figsize=(14, 8))
                    
                    # Plot average FFT across all channels
                    ax.semilogy(bin_centers, avg_fft, linewidth=2, color='blue')
                    
                    # Add title and labels
                    window_seconds = analysis_results.get("window_seconds", 10)
                    bin_width = analysis_results.get("fft_bin_width_hz", 2)
                    ax.set_title(f"Average EEG Power Spectrum ({bin_width}Hz bins, {window_seconds}s windows)")
                    ax.set_xlabel("Frequency (Hz)")
                    ax.set_ylabel("Power (log scale)")
                    ax.grid(True, which="both", ls="--", alpha=0.7)
                    
                    # Highlight 60Hz line noise region
                    ax.axvspan(59, 61, alpha=0.2, color='red', label='60Hz Line Noise')
                    
                    # Add other frequency bands of interest (delta, theta, alpha, beta, gamma)
                    ax.axvspan(0.5, 4, alpha=0.1, color='blue', label='Delta (0.5-4Hz)')
                    ax.axvspan(4, 8, alpha=0.1, color='green', label='Theta (4-8Hz)')
                    ax.axvspan(8, 13, alpha=0.1, color='orange', label='Alpha (8-13Hz)')
                    ax.axvspan(13, 30, alpha=0.1, color='purple', label='Beta (13-30Hz)')
                    
                    # Add legend
                    ax.legend(loc='upper right')
                    
                    # Set x-axis limits
                    max_freq = analysis_results.get("fft_max_freq_hz", 200)
                    ax.set_xlim(0, max_freq)
                    
                    plt.tight_layout()
                    plt.savefig(local_fft_path)
                    plt.close()
                    
                    # Add to output files
                    dest_key = f"{self.destination_prefix}{fft_plot_file_name}"
                    output_files.append((local_fft_path, dest_key))

        return output_files

    @classmethod
    def add_subclass_arguments(cls, parser):
        """Add analysis-specific command-line arguments.

        Args:
            parser: ArgumentParser instance
        """
        parser.add_argument('--no-plots', action='store_false', dest='plot_60hz_noise',
                           help='Disable generation of 60Hz noise plots')

    @classmethod
    def from_args(cls, args):
        """Create a transform instance from command-line arguments.

        Args:
            args: Parsed command-line arguments

        Returns:
            Instance of AnalysisTransform
        """
        # Extract arguments
        source_prefix = getattr(args, 'source_prefix', cls.SOURCE_PREFIX)
        dest_prefix = getattr(args, 'dest_prefix', cls.DEST_PREFIX)

        return cls(
            source_prefix=source_prefix,
            destination_prefix=dest_prefix,
            plot_60hz_noise=args.plot_60hz_noise,
            s3_bucket=args.s3_bucket,
            verbose=args.verbose,
            log_file=args.log_file,
            dry_run=args.dry_run,
            keep_local=args.keep_local
        )


# Entry point for running the transform from the command line
if __name__ == "__main__":
    AnalysisTransform.run_from_command_line()
