#!/usr/bin/env python3
"""
A simple script to extract and display original and corrected text from W group in an H5 file.
Usage: python inspect-W.py <path-to-h5-file>
"""

import sys
import h5py

def main():
    # Check command-line arguments
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <path-to-h5-file>")
        sys.exit(1)
    
    h5_path = sys.argv[1]
    
    try:
        # Open the H5 file
        with h5py.File(h5_path, 'r') as f:
            # Get original text
            original_text = f['language/W'].attrs.get('text', 'No original text found')
            
            # Get corrected text if available
            if 'W_corrected' in f['language']:
                corrected_text = f['language/W_corrected'].attrs.get('text', 'No corrected text found')
                correctness_score = f['language/W_corrected'].attrs.get('correctness_score', 0.0)
            else:
                corrected_text = 'No W_corrected group found'
                correctness_score = 0.0
            
            # Print the texts
            print("\nOriginal Text:")
            print(original_text)
            
            print("\nCorrected Text:")
            print(corrected_text)
            
            if 'W_corrected' in f['language']:
                print(f"\nCorrectness Score: {correctness_score:.2f}")
            
    except FileNotFoundError:
        print(f"Error: File not found: {h5_path}")
        sys.exit(1)
    except KeyError as e:
        print(f"Error: Missing group or dataset in H5 file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading H5 file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()