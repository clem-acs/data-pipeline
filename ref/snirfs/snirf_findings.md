# SNIRF File Format Analysis and Code Improvements

## Summary of Findings

After examining the SNIRF file structure and the current `read_snirf.py` code, we've identified several issues with the original code's assumptions:

1. **Structure mismatch**: The original code expected a nested hierarchical structure where `nirs` contains numbered elements (like `nirs[1]`, `nirs[2]`), but the actual file has a flat structure with direct children (`nirs/data1`, `nirs/probe`, etc.).

2. **Data access pattern**: The script tried to iterate through `nirs_data.data` elements, but in the actual file, data is accessed directly as `nirs['data1']`.

3. **Measurement lists**: The script expected a single `measurementList` per data element, but the actual file has thousands of individually numbered measurement lists (`measurementList1`, `measurementList2`, etc.) - one for each channel.

4. **Large data dimensions**: The MOMENTS files contain a large number of channels (4038) which requires special handling to avoid performance issues.

## Fixed Code Improvements

We've made the following improvements:

1. **Direct HDF5 access**: The updated code now directly uses `h5py` for low-level access to the SNIRF file structure, which gives more control over data handling.

2. **Flat structure support**: The code now correctly handles the flat structure found in actual SNIRF files.

3. **Efficient measurement list handling**: Instead of trying to load all measurement lists at once, the code samples a subset to avoid memory issues.

4. **Performance optimization**: For validation checks, we limit processing to a subset of channels rather than checking all 4038 channels.

5. **Error handling**: Improved error handling throughout the code to gracefully handle missing or unexpected data.

6. **Data validation**: Added checks for NaN and Inf values to help identify potential data quality issues.

7. **JSON output**: Created a clean JSON export with proper handling of NumPy types.

## Data Format Insights

The MOMENTS SNIRF file has the following characteristics:

- Format version: 1.0
- Data shape: 17064 time points × 4038 channels
- Contains Time Domain - Moments data (type 301)
- Uses 2 wavelengths: 690nm and 905nm
- Has 57 sources and 110 detectors
- No NaN or Inf values present in the data
- Each channel corresponds to a specific source-detector-wavelength-moment combination

## Utility Tools Created

1. **read_snirf.py**: Updated to correctly read and display metadata from SNIRF files.

2. **check_data_structure.py**: Created to specifically examine the structure of SNIRF files.

3. **snirf_data_extractor.py**: A complete utility class that provides clean APIs for extracting and processing SNIRF data, including:
   - Metadata extraction
   - Data time series access
   - Probe information retrieval
   - Measurement mapping
   - Channel validation
   - JSON summary export

## Next Steps

To further improve SNIRF file handling:

1. **Integration with data pipeline**: Use the new `SnirfDataExtractor` class in your data pipeline to handle SNIRF files.

2. **Channel mapping**: Create more sophisticated mapping between channels and their physiological meaning based on source-detector-wavelength combinations.

3. **Quality metrics**: Implement additional data quality metrics specific to TD-fNIRS data.

4. **Batch processing**: Add support for batch processing multiple SNIRF files.

5. **Testing suite**: Create a comprehensive test suite for the SNIRF handling code.