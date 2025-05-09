#!/usr/bin/env python3
"""Test script for unified data loader."""

import sys
import numpy as np
from backend.data_processing import load_data

if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} path/to/session.h5")
    sys.exit(1)

# Load data
data = load_data(h5_file_path=sys.argv[1])

# Print summary
print(f"Session: {data.session_id} ({data.start_time} to {data.end_time})")

# Neural data
print(f"\nNeural data: {len(data.neural_data)} devices")
for d_id, d_data in data.neural_data.items():
    frames = d_data['data'].shape[0]
    print(f"  {d_id}: {frames} frames")

# Events
print(f"\nEvents: {len(data.events)} types")
for e_type, e_data in data.events.items():
    print(f"  {e_type}: {len(e_data['data'])} events")

# Audio
if data.audio_data:
    print(f"\nAudio: sample rate {data.audio_data['sample_rate']} Hz")
    print(f"  Timestamps: {len(data.audio_data['timestamps'])} chunks")

# Language
print(f"\nLanguage data: {len(data.language_data)} subgroups")
for sg, sg_data in data.language_data.items():
    chars = len(sg_data.get('chars', []))
    words = len(sg_data.get('words', []))
    print(f"  {sg}: {chars} chars, {words} words")