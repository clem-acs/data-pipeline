# S3 H5 fNIRS Data Explorer

This utility script downloads H5 files from the curated-h5/ S3 bucket and analyzes them for fNIRS data content.

## Overview

The script performs the following:

1. Lists H5 files from the curated-h5/ S3 prefix
2. Downloads sample H5 files
3. Explores their internal structure, looking for fNIRS data
4. Analyzes and visualizes any found fNIRS data
5. Creates a detailed report comparing the structure to SNIRF format

## Requirements

- boto3
- h5py
- numpy
- matplotlib

## Usage

```bash
# Activate the virtual environment
python -m venv .venv
source .venv/bin/activate
pip install boto3 h5py numpy matplotlib

# Run the script with default settings (downloads 1 sample file)
python s3h5fnirs.py

# Download and analyze 5 H5 files
python s3h5fnirs.py --sample-count 5

# Specify a download directory
python s3h5fnirs.py --download-dir ./downloaded_files

# Use verbose logging
python s3h5fnirs.py -v
```

## Command-line Options

- `--bucket`: S3 bucket name (default: conduit-data-dev)
- `--prefix`: S3 prefix for curated H5 files (default: curated-h5/)
- `--sample-count`: Number of H5 files to sample (default: 1)
- `--download-dir`: Directory to download H5 files to (default: temp directory)
- `--verbose`, `-v`: Enable verbose logging

## Outputs

For each analyzed H5 file, the script produces:
- A markdown report with detailed analysis
- Visualizations of sample fNIRS data (if found)
- Console output summarizing the findings

## AWS Credentials

The script requires AWS credentials to be set as environment variables:
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_DEFAULT_REGION` (default: us-east-1)