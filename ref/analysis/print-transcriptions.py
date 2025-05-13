#!/usr/bin/env python3
"""
Print-Transcriptions: A simple script to extract and print transcribed words from H5 files.
Usage: python print-transcriptions.py /path/to/your/file.h5
"""

import sys
import h5py
import argparse
from datetime import datetime

def extract_transcribed_words(h5_file_path):
    """Extract and print transcribed words from the 'S' subgroup."""
    try:
        with h5py.File(h5_file_path, 'r') as f:
            # Check if file has transcription word data
            if 'language' not in f or 'S' not in f['language'] or 'words' not in f['language']['S']:
                print(f"File {h5_file_path} does not contain transcribed words (language/S/words).")
                return
            
            # Get the words dataset
            words_dataset = f['language/S/words']
            print(f"Total words in dataset: {len(words_dataset)}")
            
            # Extract words and their timestamps
            all_words = []
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
                
                all_words.append((word, start_ts, end_ts, element_id))
            
            # Sort by start timestamp to ensure correct order
            all_words.sort(key=lambda x: x[1])
            
            # Create element groups (group words by element_id)
            element_groups = {}
            for word, start_ts, end_ts, element_id in all_words:
                if element_id not in element_groups:
                    element_groups[element_id] = []
                element_groups[element_id].append((word, start_ts, end_ts))
            
            # Print by element
            print("\n===== TRANSCRIBED TEXT =====")
            
            # First print the entire transcript
            full_text = ' '.join(word[0] for word in all_words)
            print(f"FULL TRANSCRIPT: {full_text}")
            print("\n===== BY ELEMENT =====")
            
            # Then print by element
            for element_id, words in element_groups.items():
                element_text = ' '.join(word[0] for word in words)
                start_time = min(word[1] for word in words)
                end_time = max(word[2] for word in words)
                print(f"Element {element_id}: [{format_timestamp(start_time)} - {format_timestamp(end_time)}]")
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
    parser = argparse.ArgumentParser(description="Extract and print transcribed words from H5 files")
    parser.add_argument("file_path", help="Path to the H5 file")
    args = parser.parse_args()
    
    extract_transcribed_words(args.file_path)