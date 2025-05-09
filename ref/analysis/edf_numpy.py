import numpy as np
import pyedflib

def edf_to_numpy(edf_file_path, eeg_only=True, convert_to_mv=True):
    """
    Convert an EDF file to a NumPy array.

    Parameters:
    edf_file_path (str): Path to the EDF file
    eeg_only (bool): If True, return only the 32 EEG channels, otherwise return all channels
    convert_to_mv (bool): If True, convert values to millivolts (multiply by 0.001)

    Returns:
    numpy.ndarray: Array with dimensions (channels, samples)
    dict: Channel information including labels
    """
    # Open the EDF file
    f = pyedflib.EdfReader(edf_file_path)

    # Get all channel labels
    labels = f.getSignalLabels()
    
    # Define the indices of the 32 EEG channels (indices 4-35)
    eeg_indices = list(range(4, 36))  # Corresponds to standard 10-20 EEG channels
    eeg_labels = [labels[i] for i in eeg_indices]
    
    # Determine which channels to use
    if eeg_only:
        channel_indices = eeg_indices
        selected_labels = eeg_labels
    else:
        channel_indices = list(range(f.signals_in_file))
        selected_labels = labels

    # Get number of samples
    n_samples = f.getNSamples()[0]  # Assuming all channels have the same number of samples

    # Initialize the NumPy array to store the data
    data = np.zeros((len(channel_indices), n_samples))

    # Read each selected channel into the array
    for i, channel_idx in enumerate(channel_indices):
        data[i, :] = f.readSignal(channel_idx)
    
    # Convert to millivolts if requested
    if convert_to_mv:
        data *= 0.001  # Convert to mV
    
    # Close the file
    f.close()
    
    # Prepare channel info
    channel_info = {
        'labels': selected_labels,
        'indices': channel_indices
    }

    return data, channel_info

if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert EDF file to NumPy array")
    parser.add_argument("input_file", help="Input EDF file path")
    parser.add_argument("output_file", help="Output NPY file path")
    parser.add_argument("--all-channels", action="store_true", help="Include all channels, not just EEG")
    parser.add_argument("--raw-units", action="store_true", help="Keep data in raw units instead of converting to mV")
    args = parser.parse_args()

    # Convert EDF to NumPy array
    data, channel_info = edf_to_numpy(
        args.input_file, 
        eeg_only=not args.all_channels,
        convert_to_mv=not args.raw_units
    )

    # Save the NumPy array to a file
    np.save(args.output_file, data)
    
    # Save channel info to a companion file
    info_file = args.output_file.replace('.npy', '_channels.npz')
    np.savez(info_file, labels=channel_info['labels'], indices=channel_info['indices'])

    print(f"Converted {args.input_file} to NumPy array with shape {data.shape}")
    print(f"Data units: {'raw' if args.raw_units else 'mV'}")
    print(f"Channels: {'all' if args.all_channels else 'EEG only (32 channels)'}")
    print(f"Selected channels: {', '.join(channel_info['labels'])}")
    print(f"Data saved to: {args.output_file}")
    print(f"Channel info saved to: {info_file}")
