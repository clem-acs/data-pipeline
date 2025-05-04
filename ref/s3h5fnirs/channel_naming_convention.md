# fNIRS Channel Naming Convention

This document describes the naming convention for fNIRS channels in the curated H5 files. The naming convention is derived from the data processing scripts in `/ref/s3h5fnirs/`, particularly `check_channel_patterns.py` and `check_channel_transitions.py`.

## Channel Name Structure

Each channel in the dataset follows this standardized naming pattern:

```
W{wavelength_idx}({wavelength_name})_M{moment_idx}({moment_name})_S{source_module}_{source_id}_D{detector_module}_{detector_id}
```

## Naming Components

### 1. Wavelength (`W`)
Each channel measures either red or infrared light:
- `W0(Red)`: Red light wavelength
- `W1(IR)`: Infrared light wavelength

### 2. Moment (`M`)
Each channel records one of three moments of the time domain measurement:
- `M0(Zeroth)`: Zeroth moment
- `M1(First)`: First moment
- `M2(Second)`: Second moment

### 3. Source (`S`)
Each source is identified by two components:
- **Module number**: Ranges from 1 to 48
- **ID within module**: Ranges from 1 to 3

Format: `S{module}_{id}`

Example: `S12_3` refers to source in module 12, ID 3

### 4. Detector (`D`)
Each detector is identified by two components:
- **Module number**: Ranges from 1 to 48
- **ID within module**: Ranges from 1 to 6 (Note: Position 7, the central detector, is not used in measurements)

Format: `D{module}_{id}`

Example: `D24_5` refers to detector in module 24, ID 5

## Channel Index Calculation

In the dataset, channels are arranged in a specific order with a total of 248,832 channels. The index of a channel can be calculated as:

```
index = ((((wavelength_idx * 3 + moment_idx) * 48 + (source_module-1)) * 3 + (source_id-1)) * 48 + (detector_module-1)) * 6 + (detector_id-1)
```

Where:
- `wavelength_idx`: 0 (Red) or 1 (IR)
- `moment_idx`: 0 (Zeroth), 1 (First), or 2 (Second)
- `source_module`: 1-48
- `source_id`: 1-3
- `detector_module`: 1-48
- `detector_id`: 1-6

## Examples

1. `W0(Red)_M0(Zeroth)_S1_1_D1_1`
   - Red light
   - Zeroth moment
   - Source module 1, ID 1
   - Detector module 1, ID 1
   - This would be the first channel in the dataset (index 0)

2. `W1(IR)_M2(Second)_S24_3_D36_5`
   - IR light
   - Second moment
   - Source module 24, ID 3
   - Detector module 36, ID 5
   - This would have index: ((((1*3+2)*48+(24-1))*3+(3-1))*48+(36-1))*6+(5-1) = 211,240

## Data Structure in H5 Files

In the curated H5 files, the fNIRS data is located at `devices/fnirs/frames_data` with an expected shape of (num_frames, 248832, 1), where:
- First dimension: number of frames (time points)
- Second dimension: 248,832 channels
- Third dimension: singleton dimension (can be squeezed)

## Valid Channels

Not all 248,832 theoretical channels contain valid data. Many channels show `-inf` values, indicating they are invalid or unused. This typically happens because:

1. Only a subset of modules are used in measurements (approximately 20 out of 48 modules)
2. Within modules, certain source-detector combinations may not be within valid measurement range
3. The central detector (position 7) in each module is consistently excluded

If a particular source-detector combination is valid, all 6 channels associated with it (2 wavelengths × 3 moments) are typically valid. If one channel for a source-detector pair is invalid, all channels for that pair are typically invalid.

## Total Channel Count

The total number of possible channels is:
- 2 wavelengths (Red, IR)
- 3 moments (Zeroth, First, Second)
- 48 source modules
- 3 source IDs per module
- 48 detector modules
- 6 detector IDs per module

Total: 2 × 3 × 48 × 3 × 48 × 6 = 248,832 channels

However, as noted above, only a subset of these theoretical channels will contain valid measurements in any given dataset.