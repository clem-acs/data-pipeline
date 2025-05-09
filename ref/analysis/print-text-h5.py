#!/usr/bin/env python3
"""
Print-Text-H5: A simple script to extract and print reconstructed text from H5 files.
Usage: python print-text-h5.py /path/to/your/file.h5
"""

import sys
import h5py
import argparse
from datetime import datetime


def format_timestamp(ms_timestamp):
    """Convert a millisecond timestamp to a readable date string."""
    try:
        dt = datetime.fromtimestamp(ms_timestamp / 1000.0)
        return dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    except:
        return f"{ms_timestamp}"


def extract_reconstructed_text(h5_file_path):
    """Extract and print reconstructed text from an H5 file."""
    try:
        with h5py.File(h5_file_path, 'r') as f:
            # Check if file has reconstructed text
            if 'language' not in f or 'W' not in f['language'] or 'chars' not in f['language']['W']:
                print(f"File {h5_file_path} does not contain reconstructed text (language/W/chars).")
                return
            
            # Get the chars dataset
            chars_dataset = f['language/W/chars']
            print(f"Total records in dataset: {len(chars_dataset)}")
            
            # Extract characters and build full text
            all_chars = []
            
            for char_record in chars_dataset:
                # Get the character, handling bytes if needed
                char = char_record['char']
                if isinstance(char, bytes):
                    char = char.decode('utf-8')
                timestamp = char_record['timestamp']
                all_chars.append((char, timestamp))
            
            # Sort by timestamp to ensure correct order
            all_chars.sort(key=lambda x: x[1])
            
            # Combine into text
            full_text = ''.join(char[0] for char in all_chars)
            
            # Print the text
            print("\n===== RECONSTRUCTED TEXT =====")
            print(full_text)
            print("\n===== STATS =====")
            print(f"Total characters: {len(full_text)}")
            
            # Get time range
            if all_chars:
                start_time = all_chars[0][1]
                end_time = all_chars[-1][1]
                
                print(f"Start time: {format_timestamp(start_time)}")
                print(f"End time: {format_timestamp(end_time)}")
                
                duration_sec = (end_time - start_time) / 1000
                print(f"Duration: {duration_sec:.2f} seconds")
                
                if duration_sec > 0:
                    typing_speed = len(full_text) / duration_sec * 60
                    print(f"Average typing speed: {typing_speed:.2f} chars/minute")
            
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        import traceback
        print(traceback.format_exc())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and print reconstructed text from H5 files")
    parser.add_argument("file_path", help="Path to the H5 file")
    args = parser.parse_args()
    
    extract_reconstructed_text(args.file_path)