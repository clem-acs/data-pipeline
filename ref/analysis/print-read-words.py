#!/usr/bin/env python3
"""
Print-Read-Words: A simple script to extract and print words from the 'R' language subgroup.
Usage: python print-read-words.py /path/to/your/file.h5
"""

import sys
import h5py
import argparse
import json
from datetime import datetime

def extract_read_words(h5_file_path):
    """Extract and print read words from the 'R' subgroup."""
    try:
        with h5py.File(h5_file_path, 'r') as f:
            # Check if file has read word data
            if 'language' not in f or 'R' not in f['language'] or 'words' not in f['language']['R']:
                print(f"File {h5_file_path} does not contain read words (language/R/words).")
                return
            
            # Get the words dataset
            words_dataset = f['language/R/words']
            print(f"Total words in R dataset: {len(words_dataset)}")
            
            # Extract words and their timestamps, separating by mode
            all_words = []
            r_words = []
            lr_words = []
            
            for word_record in words_dataset:
                # Get the word, handling bytes if needed
                word = word_record['word']
                if isinstance(word, bytes):
                    word = word.decode('utf-8')
                
                # Get timestamps and element ID
                start_ts = word_record['start_timestamp']
                end_ts = word_record['end_timestamp']
                
                # Get element_id
                element_id = word_record['element_id']
                if isinstance(element_id, bytes):
                    element_id = element_id.decode('utf-8')
                
                # Get mode
                mode = "R"  # Default
                if 'mode' in word_record.dtype.names:
                    mode = word_record['mode']
                    if isinstance(mode, bytes):
                        mode = mode.decode('utf-8')
                
                # Get display_reference if available
                display_ref = None
                if 'display_reference' in word_record.dtype.names:
                    try:
                        display_ref_str = word_record['display_reference']
                        if isinstance(display_ref_str, bytes):
                            display_ref_str = display_ref_str.decode('utf-8')
                        if display_ref_str:
                            display_ref = json.loads(display_ref_str)
                    except:
                        pass
                
                word_info = (word, start_ts, end_ts, element_id, mode, display_ref)
                all_words.append(word_info)
                
                # Separate by mode
                if mode == "LR":
                    lr_words.append(word_info)
                else:
                    r_words.append(word_info)
            
            # Sort by start timestamp to ensure correct order
            all_words.sort(key=lambda x: x[1])
            r_words.sort(key=lambda x: x[1])
            lr_words.sort(key=lambda x: x[1])
            
            # Create element groups (group words by element_id)
            element_groups = {}
            for word, start_ts, end_ts, element_id, mode, display_ref in all_words:
                if element_id not in element_groups:
                    element_groups[element_id] = []
                element_groups[element_id].append((word, start_ts, end_ts, mode))
            
            # Print all read words
            print("\n===== ALL READ WORDS =====")
            # First print the entire text
            full_text = ' '.join(word[0] for word in all_words)
            print(f"FULL TEXT: {full_text}")
            print(f"Total read words: {len(all_words)}")
            print(f"Mode R words: {len(r_words)}")
            print(f"Mode LR words: {len(lr_words)}")
            
            # Print words by mode
            if r_words:
                print("\n===== MODE R WORDS =====")
                r_text = ' '.join(word[0] for word in r_words)
                print(f"TEXT: {r_text}")
            
            if lr_words:
                print("\n===== MODE LR WORDS =====")
                lr_text = ' '.join(word[0] for word in lr_words)
                print(f"TEXT: {lr_text}")
            
            # Print words by element
            print("\n===== WORDS BY ELEMENT =====")
            for element_id, words in element_groups.items():
                element_text = ' '.join(word[0] for word in words)
                start_time = min(word[1] for word in words)
                end_time = max(word[2] for word in words)
                modes = set(word[3] for word in words)
                
                print(f"Element {element_id}: [{format_timestamp(start_time)} - {format_timestamp(end_time)}] Modes: {', '.join(modes)}")
                print(f"  {element_text}")
                print()
                
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        import traceback
        print(traceback.format_exc())

def format_timestamp(ms_timestamp):
    """Convert a millisecond timestamp to a readable date string."""
    try:
        dt = datetime.fromtimestamp(ms_timestamp / 1000.0)
        return dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    except:
        return f"{ms_timestamp}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and print read words from H5 files")
    parser.add_argument("file_path", help="Path to the H5 file")
    args = parser.parse_args()
    
    extract_read_words(args.file_path)