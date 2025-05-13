# Transform Implementation Guide

## Overview
This guide outlines a practical, minimal approach to implementing and testing transforms for the data pipeline. It focuses on what actually works with the least amount of overhead.

## Step 1: Create a Skeleton Transform

Start with a minimal transform that helps you understand the data:

1. **Copy an existing transform**
   ```bash
   cp data-pipeline/transforms/t2D_validate_v0.py data-pipeline/transforms/t2F_my_transform_v0.py
   ```

2. **Modify the basics**:
   - Update class name and docstrings
   - Set SOURCE_PREFIX and DEST_PREFIX
   - Add simple logging to see what data is available

3. **Focus on what you need to know**:
   - Examine relevant datasets
   - Output a simple JSON report
   - Don't try to be comprehensive - focus on your task

## Step 2: Register the Transform in cli.py

```python
# In data-pipeline/cli.py
from transforms.t2F_my_transform_v0 import MyTransform

TRANSFORMS = {
    'curate': CurateTransform,
    'window': WindowTransform,
    'validate': ValidateTransform,
    'analyze': AnalysisTransform,
    'my_transform': MyTransform,  # Add your transform with a simple name
}
```

## Step 3: Run in Test Mode

This is the command that actually works:

```bash
# From the data-pipeline directory:
PYTHONPATH=. python cli.py my_transform --session SESSION_ID --test
```

The `PYTHONPATH=.` prefix is necessary to make imports work properly when running directly.

## Step 4: Look at the Output

```bash
# See what was created:
ls transform_workdir/my_transform_*/ 

# Look at the JSON:
jq '.' transform_workdir/my_transform_*/SESSION_ID_report.json
```

## Step 5: Build Incrementally

1. **Add core functionality**
   - Implement only what you need
   - Extract necessary data when the file is open
   - Process after the file is closed

2. **Test frequently**
   - Run with `--test` after each change
   - Fix issues as they arise
   - Don't wait until the end to test

3. **Add features one by one**
   - Get the basics working first
   - Add more complex features later
   - Keep the workflow iterative

## Step 6: Avoid Common Problems

1. **H5 file handling**
   - Get all data you need while the file is open
   ```python
   with h5py.File(file_path, 'r') as f:
       audio_data = f['/audio/audio_data'][:]  # Extract data
       timestamps = f['/audio/timestamps'][:]  # Extract data
   
   # Process outside the context
   process_data(audio_data, timestamps)
   ```

2. **File paths**
   - Use correct paths for source and destination files
   ```python
   dest_key = f"{self.destination_prefix}{file_name}"
   ```

## Step 7: Test Different Sessions

```bash
# Try more than one session:
PYTHONPATH=. python cli.py my_transform --session SESSION_ID1 --test
PYTHONPATH=. python cli.py my_transform --session SESSION_ID2 --test
```

## Real-World Implementation Process

1. **Create a skeleton**
   - Make it log structure information
   - Don't worry about processing yet

2. **Add to cli.py**
   - Register with a simple name
   - Use the test mode to see output

3. **Look at the data**
   - Check the structure
   - See what's available 

4. **Extract what you need**
   - Get just the data you need
   - Keep it minimal

5. **Process step by step**
   - Build processing logic incrementally
   - Test each step

6. **Output the results**
   - Create output files
   - Generate a report

7. **Try with different sessions**
   - Make sure it works in different cases
   - Fix any issues that arise

## Example: Audio Extraction

```python
def process_session(self, session: Session) -> Dict:
    session_id = session.session_id
    
    # Download the h5 file
    local_h5_path = session.download_file(f"{self.source_prefix}{session_id}.h5")
    
    # Extract data from the file
    with h5py.File(local_h5_path, 'r') as f:
        # Get what we need
        audio_data = f['/audio/audio_data'][:]
        audio_ts = f['/audio/timestamps'][:]
        rec_start = f['/events/recording_start/timestamps'][:]
        rec_stop = f['/events/recording_stop/timestamps'][:]
    
    # Process data (after file is closed)
    sample_rate = 16000  # Assume or calculate
    segments = []
    
    for i in range(len(rec_start)):
        if i >= len(rec_stop):
            break  # No matching stop event
        
        # Use client-side timestamps (index 1)
        start_ts = rec_start[i][1]
        stop_ts = rec_stop[i][1]
        
        # Extract audio chunks in this range
        # [implementation details...]
        
        # Save as WAV file
        wav_filename = f"{session_id}_segment_{i+1}.wav"
        local_wav_path = session.create_upload_file(wav_filename)
        scipy.io.wavfile.write(local_wav_path, sample_rate, segment_data)
        
        segments.append((local_wav_path, f"{self.destination_prefix}{wav_filename}"))
    
    # Create report
    report_filename = f"{session_id}_report.json"
    local_report_path = session.create_upload_file(report_filename)
    
    with open(local_report_path, 'w') as f:
        json.dump({
            "session_id": session_id,
            "segments": len(segments),
            "sample_rate": sample_rate
        }, f, indent=2)
    
    # Return all files to upload
    return {
        "status": "success",
        "metadata": {"segments": len(segments)},
        "files_to_copy": [],
        "files_to_upload": [(local_report_path, f"{self.destination_prefix}{report_filename}")] + segments
    }
```

## Simple Checklist

1. ☑ Copy and modify an existing transform
2. ☑ Add to cli.py with a simple name
3. ☑ Run with `PYTHONPATH=. python cli.py my_transform --session ID --test` 
4. ☑ Look at the output and logs
5. ☑ Extract needed data while file is open
6. ☑ Process data after file is closed
7. ☑ Save outputs and report
8. ☑ Test with different sessions
9. ☑ Fix any issues that arise