# SNIRF File Reader

This project contains a sloppy bunch of scripts to explore SNIRF (Shared Near Infrared Format) files and understand how to read them. These scripts are for exploration purposes only and not intended for production use.

## Environment Setup

This project uses a nested requirements structure - the dependencies for reading SNIRF files are installed in a local virtual environment:

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install mne h5py pysnirf2 matplotlib pandas
```

## Scripts

### 1. MNE-based SNIRF Reader (`read_snirf_metadata.py`)

Uses the MNE package to read SNIRF files. Note: MNE only supports reading continuous wave (CW) and processed hemoglobin data types (type codes 1 and 99999).

### 2. PySNIRF2-based Reader (`read_snirf_with_pysnirf2.py`)

Uses the pysnirf2 package to read all types of SNIRF files, including Time Domain (TD) data that MNE doesn't support.

### 3. H5py Exploration (`explore_snirf_with_h5py.py`)

Directly accesses the underlying HDF5 structure of SNIRF files to explore their complete structure.

### 4. SNIRF Comparison (`compare_snirf_files.py`)

Compares key metrics across multiple SNIRF files to understand their differences.

### 5. Data Type Analysis (`examine_data_types.py`)

Analyzes the different data types found in SNIRF files and their properties.

### 6. Data Visualization (`visualize_snirf_data.py`)

Creates visualizations to compare the content of different SNIRF files.

## Key Findings

Our analysis revealed that the three sample SNIRF files represent different stages of the same fNIRS data:

1. **GATES file** (4.38 GB): Contains raw time-domain histogram data (dataType 201) with full photon counts across 33,650 channels.

2. **MOMENTS file** (526 MB): Contains statistical moments (dataType 301) calculated from time-domain data with 4,038 channels.

3. **HB_MOMENTS file** (175 MB): Contains processed hemoglobin concentration data (dataType 99999) with 1,346 channels.

All files share the same time axis (17,064 time points at 4.76 Hz) and probe configuration (57 sources, 110 detectors, 2 wavelengths: 690nm and 905nm).

The detailed findings are available in the `snirf_analysis_summary.md` file.

## Data Types in SNIRF Files

SNIRF files support different fNIRS modalities, categorized with specific type codes:

- 001-100: Raw - Continuous Wave (CW)
- 101-200: Raw - Frequency Domain (FD)
- 201-300: Raw - Time Domain - Gated (TD Gated)
- 301-400: Raw - Time Domain - Moments (TD Moments)
- 401-500: Raw - Diffuse Correlation Spectroscopy (DCS)
- 99999: Processed Data (e.g., hemoglobin concentration)

## Note on Package Compatibility

The `pysnirf2` package requires NumPy < 2.0 due to usage of deprecated NumPy string types. The library is now maintained as the `snirf` package and should be preferred for new projects.