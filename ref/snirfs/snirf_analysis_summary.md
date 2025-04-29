# SNIRF Files Analysis Summary

## Overview of the Three SNIRF File Types

We analyzed three SNIRF files from the same dataset:

1. **test_Test_6d1af09_HB_MOMENTS.snirf**
   - Contains processed hemoglobin concentration data (HbO)
   - Data Type: 99999 (HbO)
   - Size: 175.23 MB
   - Shape: (17064, 1346) - 17,064 time points, 1,346 channels
   - Values range: -51.85 to 41.49 μM (micromolar)

2. **test_Test_6d1af09_MOMENTS.snirf**
   - Contains time-domain moments data
   - Data Type: 301 (Time Domain - Moments - Amplitude)
   - Size: 525.70 MB
   - Shape: (17064, 4038) - 17,064 time points, 4,038 channels
   - Values range: ~14.8M to ~16.5M (counts)

3. **test_Test_6d1af09_GATES.snirf**
   - Contains full time-domain histogram data
   - Data Type: 201 (Time Domain - Full Histogram - Amplitude)
   - Size: 4,380.83 MB (4.3 GB)
   - Shape: (17064, 33650) - 17,064 time points, 33,650 channels
   - Values range: ~7.4M to ~8.3M (counts)

## Key Findings

### Common Structure
- All files follow SNIRF format version 1.0
- All share the same time axis: 17,064 time points at ~4.76 Hz (0.21s interval)
- All have identical metadata and probe configuration with:
  - 57 sources
  - 110 detectors
  - 2 wavelengths: 690nm and 905nm

### Differences
- **Data Type**: Each file contains a different representation of the same underlying data
  - HB_MOMENTS: Processed hemoglobin concentrations
  - MOMENTS: Time-domain moments (statistical description of time-resolved data)
  - GATES: Full time-domain histogram data (raw measurement data)
  
- **Data Volume**: Huge difference in data volume due to the level of processing
  - GATES file is ~25x larger than HB_MOMENTS file
  - GATES contains the most detailed raw data
  - MOMENTS is an intermediate representation (statistical moments)
  - HB_MOMENTS is the most processed (calculated hemoglobin concentrations)

- **Measurements**: Different number of measurements per source-detector pair
  - HB_MOMENTS: 1,346 measurements
  - MOMENTS: 4,038 measurements
  - GATES: 33,650 measurements

### Correlation Analysis
- MOMENTS and GATES data are highly correlated (0.90)
- HB_MOMENTS shows moderate correlation with others (0.31-0.42)
- This indicates that MOMENTS and GATES contain similar information, while HB_MOMENTS represents a different transformation of the data

## Interpretation

The three files represent different stages of fNIRS data processing:

1. **GATES file** contains the raw time-domain histogram data, showing the full photon counts over time for each source-detector pair at each wavelength. This is the most detailed but least processed representation.

2. **MOMENTS file** contains statistical moments calculated from the time-domain data, which summarizes the photon time-of-flight distributions. This is a more compact representation derived from the GATES data.

3. **HB_MOMENTS file** contains processed hemoglobin concentration values (HbO) calculated from the optical measurements, representing the final output of the processing pipeline.

The substantial size difference between files reflects the data reduction that occurs during processing, going from raw photon counts to derived hemoglobin values.

## Auxiliary Data
All files contain extensive auxiliary data including:
- EEG measurements (various channels)
- Temperature sensors (multiple locations)
- Physiological measurements

## Time Information
- Recording duration: ~3583 seconds (59.7 minutes)
- Sampling rate: 4.76 Hz
- Time unit: seconds