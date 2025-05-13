#!/usr/bin/env python3
"""
Print-Heard-Chars: A simple script to extract and print audio-aligned characters from the 'L' language subgroup.
Usage: python print-heard-chars.py /path/to/your/file.h5
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

def extract_heard_chars(h5_file_path):
    """Extract and print audio-aligned characters from the 'L' subgroup."""
    try:
        with h5py.File(h5_file_path, 'r') as f:
            # Check if file has 'L' character data
            if 'language' not in f or 'L' not in f['language'] or 'chars' not in f['language']['L']:
                print(f"File {h5_file_path} does not contain audio-aligned characters (language/L/chars).")
                return
            
            # Get the chars dataset
            chars_dataset = f['language/L/chars']
            print(f"Total characters in L dataset: {len(chars_dataset)}")
            
            # Extract characters and their timestamps
            all_chars = []
            
            for char_record in chars_dataset:
                # Get the character, handling bytes if needed
                char = char_record['char']
                if isinstance(char, bytes):
                    char = char.decode('utf-8')
                
                # Get timestamps and element ID
                start_ts = char_record['start_timestamp']
                end_ts = char_record['end_timestamp']
                
                # Get element_id
                element_id = char_record['element_id']
                if isinstance(element_id, bytes):
                    element_id = element_id.decode('utf-8')
                
                all_chars.append((char, start_ts, end_ts, element_id))
            
            # Sort by start timestamp to ensure correct order
            all_chars.sort(key=lambda x: x[1])
            
            # Create element groups (group characters by element_id)
            element_groups = {}
            for char, start_ts, end_ts, element_id in all_chars:
                if element_id not in element_groups:
                    element_groups[element_id] = []
                element_groups[element_id].append((char, start_ts, end_ts))
            
            # Print all heard characters
            print("\n===== ALL HEARD CHARACTERS =====")
            # First print the entire text
            full_text = ''.join(char[0] for char in all_chars)
            print(f"FULL TEXT: {full_text}")
            print(f"Total heard characters: {len(all_chars)}")
            
            # Print characters by element
            print("\n===== CHARACTERS BY ELEMENT =====")
            for element_id, chars in element_groups.items():
                element_text = ''.join(char[0] for char in chars)
                start_time = min(char[1] for char in chars)
                end_time = max(char[2] for char in chars)
                
                print(f"Element {element_id}: [{format_timestamp(start_time)} - {format_timestamp(end_time)}]")
                print(f"  {element_text}")
                
                # Print character-by-character timing for detailed analysis
                # Print timing for all elements, regardless of length
                print("  Character timing:")
                for char, start_ts, end_ts in chars:
                    duration_ms = end_ts - start_ts
                    print(f"    '{char}': {format_timestamp(start_ts)} - {format_timestamp(end_ts)} ({duration_ms:.0f}ms)")
                print()
                
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and print audio-aligned characters from H5 files")
    parser.add_argument("file_path", help="Path to the H5 file")
    args = parser.parse_args()
    
    extract_heard_chars(args.file_path)