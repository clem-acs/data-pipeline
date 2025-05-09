# QUICK START EXAMPLE:

```bash
cd /Users/clem/Desktop/code/data-collector/analysis && python inspect_new_sessions.py --list-only --test --filter "20250426" --min-duration 3
```

Finds test sessions on 26th april 2025 that are at least 3 minutes long

example output:

```bash
Available sessions matching criteria:
Session Name                        Start Time           End Time             Duration        Files
-----------------------------------------------------------------------------------------------
test_20250426_074220                2025-04-26 07:42:25  2025-04-26 07:54:56  12m 31s         4
asyncio_breaks_20250426_090404      2025-04-26 09:04:05  2025-04-26 09:11:00  6m 54s          3

Total sessions with duration > 3 minutes: 2
```

Then you can run the tools, with the session name in question

# Timestamp Analysis Tools

This directory contains utilities for analyzing timestamp data from various neuroimaging sessions. The scripts access data from the S3 bucket `conduit-data-dev` and work with EEG, fNIRS, and audio data.

## Prerequisites

These scripts require AWS credentials in an `env.secrets` file located in the parent directory. The file should contain:

```
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=your_region
```

## Available Scripts

### plot_combined_timestamps.py

Creates a combined plot showing timestamp offsets across all modalities (EEG, fNIRS, audio).

```bash
python plot_combined_timestamps.py --session SESSION_NAME [--exact-name] [--output-prefix PREFIX] [--test]
```

- Analyzes sessions from: `s3://conduit-data-dev/data-collector/new-sessions` (unless --test specified)
- Generates: `combined_offsets.png`
- Shows timestamp offsets across all modalities in a single visualization

### analyze_fnirs_timestamps.py

Specifically analyzes fNIRS timestamp differences between adjacent frames.

```bash
python analyze_fnirs_timestamps.py --session SESSION_NAME [--exact-name] [--output-prefix PREFIX] [--test]
```

- Analyzes sessions from: `s3://conduit-data-dev/data-collector/new-sessions/` (unless --test specified)
- Generates:
  - `fnirs_timestamp_diffs.png`: Plot of timestamp differences over time
  - `fnirs_timestamp_diffs_histogram.png`: Distribution of timestamp differences
  - `fnirs_timestamp_diffs_zoom.png`: Zoomed view of first 100 samples

### inspect_new_sessions.py

Interactive tool to list and explore session data.

```bash
# List all sessions with duration > 10 minutes
python inspect_new_sessions.py --list-only [--min-duration MINUTES] [--filter PATTERN] [--test]

# Inspect specific session
python inspect_new_sessions.py --session SESSION_NAME
```

- Analyzes sessions from: `s3://conduit-data-dev/data-collector/new-sessions/` (unless --test specified)
- Allows browsing sessions and examining their H5 file structure

## Examples

Analyze timestamps from a specific session:

```bash
python plot_combined_timestamps.py --session double_buffer_20250425_192301 --test
```

List all available sessions:

```bash
python inspect_new_sessions.py --list-only
```

## Note on Relative Paths

```bash
# From analysis directory:
python plot_combined_timestamps.py --session SESSION_NAME
```

Make sure to run from analysis/ directory, because the scripts expect `env.secrets` to be in the parent directory of where the script is located.
